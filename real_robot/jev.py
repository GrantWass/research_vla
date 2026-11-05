"""JEV as a verifier around pi0.5: watch the episode, judge grasp and place
from the cameras, and tell the runner to stop and retry when it fails.

pi0.5 drives every motion. JEV (autotrust JEV-9B, System 1:
POST /v1/decide -> calibrated yes/no) only answers questions at the two
moments a pick-and-place goes wrong:

  grasp  the gripper closed and the arm lifted: is the object in the gripper?
  place  the gripper opened again: is the object in the container?

Why a verifier and not a controller: JEV's own robot-arm demo reports 0/10
when it picks motor commands directly, and it works as yes/no visual
servoing only with an overhead camera and a calibrated planar gantry, which
the Piper rig does not have. Yes/no questions are what its decision head is
calibrated for, and each one costs ~0.2-0.4 s off the control loop.

The moments come from the gripper width alone (real units, m):
  closed -> opens wide (> OPEN_M) -> closes and stalls  = grasp attempt
  stalled width rises by RELEASE_M                      = release
"""
from __future__ import annotations

import base64
import io
import threading
import time
from dataclasses import dataclass, field

import numpy as np

OPEN_M = 0.080      # gripper wider than this: getting ready to grasp
STALL_M = 0.0015    # width change below this between two feeds: stalled
RELEASE_M = 0.0035  # stalled width grows by this: releasing
EMPTY_M = 0.020     # stalled narrower than this: closed on nothing


def data_url(img: np.ndarray, width: int = 448, quality: int = 85) -> str:
    """HxWx3 uint8 RGB -> downscaled JPEG data URL."""
    from PIL import Image

    im = Image.fromarray(img)
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


class JevClient:
    """System 1 yes/no over the robot's cameras."""

    def __init__(self, url: str = "http://localhost:8000", timeout_s: float = 30.0,
                 cameras: tuple[str, ...] = ("cam_wrist", "cam_high"), session=None):
        import requests

        self.url = url.rstrip("/") + "/v1/decide"
        self.timeout_s = timeout_s
        self.cameras = cameras
        self.http = session or requests.Session()

    def state(self, images: dict[str, np.ndarray], task: str, gripper_m: float | None) -> list:
        names = {"cam_wrist": "Wrist camera (mounted on the gripper, looking past the fingers):",
                 "cam_high": "Side camera (shows the whole table and the arm):"}
        s: list = []
        for c in self.cameras:
            if c in images:
                s += [("\n" if s else "") + names.get(c, f"{c}:"), {"image": data_url(images[c])}]
        s.append(f"\nRobot task: {task}." +
                 (f"\nGripper opening: {gripper_m * 1000:.0f} mm." if gripper_m is not None else ""))
        return s

    def yes(self, images: dict[str, np.ndarray], question: str, task: str,
            gripper_m: float | None = None) -> tuple[float, float]:
        """P(yes) for question about the current frames, and seconds taken."""
        t0 = time.perf_counter()
        r = self.http.post(self.url, json={"kind": "noul", "state": self.state(images, task, gripper_m),
                                           "question": question, "options": ["false", "true"],
                                           "thinking": "off"}, timeout=self.timeout_s)
        r.raise_for_status()
        p = r.json()["probabilities"]
        return float(p[1]), time.perf_counter() - t0


def questions(obj: str, container: str) -> dict[str, str]:
    return {"holding": f"Is the {obj} held between the robot gripper's fingers right now "
                       f"(lifted off the table, not lying on the table)?",
            "placed": f"Is the {obj} inside the {container} right now?"}


@dataclass
class Event:
    t: float
    step: int
    kind: str
    detail: dict = field(default_factory=dict)


class JevMonitor:
    """Feed it every observation (runner.run(observer=...)); it sets
    stop_event with a verdict when the attempt is decided.

    verdict: "success", "missed_grasp", "dropped", "missed_place" or None.
    Queries run in a worker thread so the control loop never waits on JEV.
    """

    def __init__(self, client, task: str, obj: str, container: str, gripper_idx: int = 6,
                 grasp_thr: float = 0.35, place_thr: float = 0.6, lift_s: float = 1.2,
                 settle_s: float = 1.0, recheck_s: float = 2.0, gripper_text: bool = True,
                 empty_check: bool = False, holding_width: float | None = None, log=print):
        self.client, self.task, self.gi, self.log = client, task, gripper_idx, log
        self.q = questions(obj, container)
        self.grasp_thr, self.place_thr = grasp_thr, place_thr
        self.lift_s, self.settle_s, self.recheck_s = lift_s, settle_s, recheck_s
        self.gripper_text, self.empty_check = gripper_text, empty_check
        self.stop_event = threading.Event()
        self.verdict: str | None = None
        self.events: list[Event] = []
        self.phase = "approach"  # approach -> open -> holding -> released
        self._prev_g: float | None = None
        self._stall_g = 0.0
        if holding_width is not None:  # attempt starts with the object already in the gripper
            self.phase, self._stall_g = "holding", holding_width
        self._due: tuple[str, float] | None = None  # (question, at time)
        self._busy = False
        self._lock = threading.Lock()

    # -- called from the control loop -------------------------------------------------
    def __call__(self, images: dict[str, np.ndarray], q_real: np.ndarray, step: int) -> None:
        g = float(q_real[self.gi])
        now = time.time()
        prev, self._prev_g = self._prev_g, g
        if self.stop_event.is_set():
            return
        if self.phase == "approach" and g > OPEN_M:
            self._event(step, "opened", g=g)
            self.phase = "open"
        elif (self.phase == "open" and prev is not None and g < OPEN_M - 0.005
              and abs(g - prev) < STALL_M):
            self._stall_g = g
            self.phase = "holding"
            self._event(step, "closed", g=g)
            if self.empty_check and g < EMPTY_M:
                return self._decide(step, "missed_grasp", why="gripper closed on nothing")
            self._due = ("holding", now + self.lift_s)
        elif self.phase == "holding" and g > self._stall_g + RELEASE_M:
            self.phase = "released"
            self._event(step, "released", g=g)
            self._due = ("placed", now + self.settle_s)
        with self._lock:
            if self._due and now >= self._due[1] and not self._busy:
                name = self._due[0]
                self._due = None
                self._busy = True
                threading.Thread(target=self._ask, args=(name, images, g, step), daemon=True).start()

    # -- worker -------------------------------------------------------------------------
    def _ask(self, name: str, images, g: float, step: int) -> None:
        try:
            p, dt = self.client.yes(images, self.q[name], self.task, g if self.gripper_text else None)
        except Exception as e:  # JEV down: never stop the robot because of the judge
            self._event(step, "jev_error", question=name, error=f"{type(e).__name__}: {e}")
            with self._lock:
                self._busy = False
            return
        self._event(step, "jev", question=name, p_yes=round(p, 4), seconds=round(dt, 3), phase=self.phase)
        with self._lock:
            self._busy = False
            if self.stop_event.is_set():
                return
            if name == "holding":
                if self.phase != "holding":
                    return  # released meanwhile; the place check decides
                if p < self.grasp_thr:
                    self._decide(step, "missed_grasp" if not self._seen_hold() else "dropped", p_yes=p)
                else:
                    self._due = ("holding", time.time() + self.recheck_s)  # catch drops in transit
            elif name == "placed":
                if p >= self.place_thr:
                    self._decide(step, "success", p_yes=p)
                else:
                    self._decide(step, "missed_place", p_yes=p)

    def final_check(self, images, q_real: np.ndarray, step: int) -> str:
        """End of attempt without a verdict (step budget ran out): ask once."""
        if self.verdict:
            return self.verdict
        p, dt = self.client.yes(images, self.q["placed"], self.task,
                                float(q_real[self.gi]) if self.gripper_text else None)
        self._event(step, "jev", question="placed", p_yes=round(p, 4), seconds=round(dt, 3), phase="final")
        self._decide(step, "success" if p >= self.place_thr else "timeout", p_yes=p)
        return self.verdict

    def _seen_hold(self) -> bool:
        return any(e.kind == "jev" and e.detail.get("question") == "holding"
                   and e.detail["p_yes"] >= self.grasp_thr for e in self.events)

    def _decide(self, step: int, verdict: str, **detail) -> None:
        self.verdict = verdict
        self._event(step, "verdict", verdict=verdict, **detail)
        self.stop_event.set()

    def _event(self, step: int, kind: str, **detail) -> None:
        e = Event(time.time(), step, kind, detail)
        self.events.append(e)
        self.log(f"[jev] step {step} {kind} {detail}")
