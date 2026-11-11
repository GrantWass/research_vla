"""JEV-9B as the grasp-offset model: three 7-way questions (reach, sideways, height).

Same interface as jevlite_offset.GraspOffset, so `jev_guard --recover align
--offset-model jev` swaps it in. The JEV server must serve the 3-D adapter
(outputs/jev_lora_3d, stripped with scripts/strip_adapter.py). The prompt matches
real_robot.jev_lora_3d training: wrist then side camera, no task line.
"""
from __future__ import annotations

import time

import numpy as np

from .jev import data_url
from .jev_lora_3d import OPTIONS, SIDE_TXT, WRIST_TXT, question
from .jevlite_offset import AXES, BUCKETS, from_arm_frame

CENTRES = np.array([-0.075, -0.045, -0.02, 0.0, 0.02, 0.045, 0.075])  # m, as in jev_lora_3d.evaluate


class JevGraspOffset:
    def __init__(self, url: str = "http://localhost:8000", obj: str = "Rubik's cube",
                 timeout_s: float = 30.0, session=None):
        import requests

        self.url = url.rstrip("/") + "/v1/decide"
        self.obj = obj
        self.timeout_s = timeout_s
        self.http = session or requests.Session()
        self.answer: dict | None = None
        self.spread = 0.0  # one deterministic model: no ensemble spread
        self.latency_s = 0.0

    def state(self, images: dict) -> list:
        return [WRIST_TXT, {"image": data_url(images["cam_wrist"])},
                SIDE_TXT, {"image": data_url(images["cam_high"])}]

    def probs(self, images: dict) -> np.ndarray:
        t0 = time.perf_counter()
        st = self.state(images)
        P = []
        for ax in AXES:
            r = self.http.post(self.url, json={"kind": "choice", "state": st, "question": question(self.obj, ax),
                                               "options": OPTIONS[ax], "thinking": "off"}, timeout=self.timeout_s)
            r.raise_for_status()
            P.append(r.json()["probabilities"])
        self.latency_s = time.perf_counter() - t0
        return np.asarray(P, float)

    def rtz(self, images: dict) -> np.ndarray:
        P = self.probs(images)
        P /= P.sum(1, keepdims=True)
        self.answer = {ax: {"bucket": BUCKETS[int(P[a].argmax())], "p": round(float(P[a].max()), 3)}
                       for a, ax in enumerate(AXES)}
        return (P * CENTRES).sum(1)

    def __call__(self, images: dict, p_xyz: np.ndarray) -> np.ndarray:
        return from_arm_frame(np.asarray(p_xyz, float), self.rtz(images))
