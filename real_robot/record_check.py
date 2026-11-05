"""Pre-flight check for demo recording (far arm leads, close arm follows): never moves the follower.

Exercises everything `make record` uses, with real hardware and data:

  can       both CAN links up
  follower  close arm read-only (passive: motors untouched): joints + gripper
            (observation.state) and end pose (observation.end_pose) finite,
            inside the limits, no arm error
  cameras   every config camera delivers fresh frames at ~fps, not blank or
            frozen; saves one JPG each
  leader    far arm in teaching mode (0xFA): you move it by hand for
            --leader-seconds; its broadcast (the action) must arrive, change
            and look sane; then back to normal mode (0xFC). Skip: --skip-leader
  dataset   --seconds of real frames at --fps into a throwaway LeRobot
            dataset (same features and writer as recorder.py), saved,
            finalized, reloaded, video frames decoded; deleted unless --keep

Run with the LeRobot venv (make record-check). Writes report.json and the
camera JPGs to --out.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

from .config import load_config
from .safety import SafetyAbort, SafetyFilter
from .verify import Report


def check_can(rep: Report, cans) -> None:
    for can in cans:
        out = subprocess.run(["ip", "-br", "link", "show", can], capture_output=True, text=True).stdout
        rep.check("can", can, " UP " in f" {out} ", out.strip() or "missing")


def check_follower(rep: Report, robot, cfg) -> None:
    q = robot.get_joint_state()
    rep.check("follower", "joints+gripper finite", bool(np.all(np.isfinite(q))),
              f"deg {np.degrees(q[:6]).round(1).tolist()}, gripper {q[6] * 1000:.1f} mm")
    try:
        SafetyFilter(cfg).check_state(q)
        rep.check("follower", "inside joint limits", True)
    except SafetyAbort as e:
        rep.check("follower", "inside joint limits", False, str(e))
    pose = robot.get_end_pose()
    rep.check("follower", "end pose finite", bool(np.all(np.isfinite(pose))) and np.any(pose != 0),
              f"xyz mm {(pose[:3] * 1000).round(0).tolist()}")
    for side, arm in robot.arms.items():
        st = arm.GetArmStatus().arm_status
        rep.check("follower", f"{side} status", int(st.err_code) == 0 and int(st.arm_status) == 0,
                  f"ctrl {st.ctrl_mode}, status {st.arm_status}, err {st.err_code}")


def check_cameras(rep: Report, robot, cfg, out: str, fps: int) -> None:
    import cv2

    t0 = time.monotonic()
    imgs = robot.get_images()  # waits for every camera's first frame
    rep.check("cameras", "first frames", True, f"{time.monotonic() - t0:.1f} s for {list(imgs)}")
    cams = robot.cameras
    start = dict(cams.frame_counts)
    time.sleep(3.0)
    imgs2 = robot.get_images()
    for name in cfg.camera_names:
        rate = (cams.frame_counts[name] - start[name]) / 3.0
        rep.check("cameras", f"{name} rate", rate >= 0.9 * fps, f"{rate:.1f} fps (want ~{fps})")
        img = imgs2[name]
        rep.check("cameras", f"{name} image", img.shape[2] == 3 and img.std() > 5,
                  f"{img.shape}, mean {img.mean():.0f}, std {img.std():.0f}")
        rep.check("cameras", f"{name} not frozen", not np.array_equal(imgs[name], img))
        path = os.path.join(out, f"{name}.jpg")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rep.check("cameras", f"{name} saved", None, path)


def check_leader(rep: Report, leader, cfg, seconds: float):
    """Teaching-mode leader: its broadcast (the recorded action) vs its own joint feedback.

    The broadcast (0x155-0x157, gripper 0x159) only comes while the arm is
    moved; the feedback (0x2A5-0x2A7, gripper 0x2A8) is always on at 200 Hz.
    Logs both over the window and checks the broadcast keeps up with the arm.
    """
    leader.connect()  # teaching mode (0xFA): limp, broadcasts while moved
    try:
        print(f"\n>>> LEADER CHECK: move the far arm by hand (every joint a little, "
              f"and open/close its gripper) for {seconds:.0f} s. Waiting for it to move...")
        leader.wait_for_broadcast(timeout_s=60.0)
        bc, fb, t_end, last = [], [], time.monotonic() + seconds, None
        while time.monotonic() < t_end:
            now = time.monotonic()
            stamp = leader.arm.GetArmJointCtrl().time_stamp
            if stamp != last:
                last = stamp
                bc.append((now, *leader.read()))
            fb.append((now, *leader.read_feedback()))
            time.sleep(0.005)
        bc, fb = np.asarray(bc), np.asarray(fb)
        rep.check("leader", "broadcast", len(bc) > 10, f"{len(bc)} updates in {seconds:.0f} s")
        if len(bc) <= 10:
            return leader.read()
        rng = np.degrees(fb[:, 1:7].max(0) - fb[:, 1:7].min(0))
        rep.check("leader", "joints moved", bool(np.all(rng > 2)),
                  f"range deg {rng.round(0).tolist()} (want > 2 on each)")
        # What the recorder would record at each instant: the newest broadcast.
        idx = np.searchsorted(bc[:, 0], fb[:, 0], side="right") - 1
        ok = idx >= 0
        held, actual, t = bc[idx[ok], 1:8], fb[ok, 1:8], fb[ok, 0]
        err = np.degrees(np.abs(held[:, :6] - actual[:, :6])).max(1)
        speed = np.degrees(np.abs(np.gradient(actual[:, :6], t, axis=0))).max(1)
        moving = speed > 5
        age = t - bc[idx[ok], 0]
        rep.check("leader", "action vs arm (moving)", float(np.percentile(err[moving], 95)) < 5 if moving.any() else None,
                  (f"newest broadcast vs the arm's own joints while moving: median {np.median(err[moving]):.1f} deg, "
                   f"p95 {np.percentile(err[moving], 95):.1f}, max {err[moving].max():.1f}" if moving.any() else "arm never moved > 5 deg/s"))
        stale = moving & (age > 0.1)
        rep.check("leader", "broadcast while moving", float(stale.mean()) < 0.02 if moving.any() else None,
                  f"arm moving but no broadcast for > 100 ms: {stale.mean() * 100:.1f}% of moving time "
                  f"(broadcast {len(bc) / seconds:.0f} Hz; it stops while the arm is held still)")
        rep.check("leader", "action vs arm (end)", None, f"{err[-1]:.1f} deg, last broadcast {age[-1]:.1f} s old")
        g_bc, g_fb = bc[:, 7] * 1000, fb[:, 7] * 1000
        rep.check("leader", "gripper", (g_bc.max() - g_bc.min()) > 5,
                  f"broadcast {g_bc.min():.0f}..{g_bc.max():.0f} mm, measured {g_fb.min():.0f}..{g_fb.max():.0f} mm")
        clipped = np.mean(np.any((bc[:, 1:8] < cfg.joint_lower) | (bc[:, 1:8] > cfg.joint_upper), axis=1))
        rep.check("leader", "inside follower limits", None,
                  f"{clipped * 100:.0f}% of broadcasts outside (recorder clips the action)")
        np.savez(os.path.join(rep.out, "leader_streams.npz"), broadcast=bc, feedback=fb)
        return leader.read()
    finally:
        leader.restore_follower_mode()
        rep.check("leader", "restored normal mode (0xFC)", None, "far arm holds where it is, motors on")


def check_dataset(rep: Report, robot, cfg, seconds: float, fps: int, keep: str | None, action) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .recorder import create_dataset

    root = keep or os.path.join(tempfile.mkdtemp(prefix="record_check_"), "ds")
    try:
        ds = create_dataset(root, cfg, fps)
        n, t0, late = int(seconds * fps), time.monotonic(), 0
        for i in range(n):
            due = t0 + i / fps
            if time.monotonic() > due + 1.0 / fps:
                late += 1
            time.sleep(max(0.0, due - time.monotonic()))
            state = robot.get_joint_state()
            act = action if action is not None else state
            frame = {"observation.state": state.astype(np.float32),
                     "observation.end_pose": robot.get_end_pose().astype(np.float32),
                     "action": np.clip(act, cfg.joint_lower, cfg.joint_upper).astype(np.float32),
                     "task": "record check"}
            for cam, img in robot.get_images().items():
                frame[f"observation.images.{cam}"] = img
            ds.add_frame(frame)
        rate = n / (time.monotonic() - t0)
        rep.check("dataset", "frame rate", rate >= 0.95 * fps and late < n * 0.05,
                  f"{rate:.1f} fps, {late} late frames of {n}")
        t1 = time.monotonic()
        ds.save_episode()
        ds.finalize()
        rep.check("dataset", "save + finalize", True, f"{time.monotonic() - t1:.1f} s for {n} frames")
        back = LeRobotDataset(ds.repo_id, root=root)
        rep.check("dataset", "reload", back.num_frames == n and back.num_episodes == 1,
                  f"{back.num_episodes} episode, {back.num_frames} frames")
        for k in (0, n - 1):
            item = back[k]
            shapes = {c: tuple(item[f"observation.images.{c}"].shape) for c in cfg.camera_names}
            rep.check("dataset", f"frame {k} decodes", all(len(v) == 3 for v in shapes.values()),
                      f"{shapes}, state {np.round(item['observation.state'].numpy()[:3], 3).tolist()}...")
        rep.check("dataset", "features", None, sorted(back.features))
    finally:
        if keep is None:
            shutil.rmtree(os.path.dirname(root), ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--follower-config", default="real_robot/configs/piper_1.json")
    ap.add_argument("--leader", default="can_piper2")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=3.0, help="dataset check length")
    ap.add_argument("--leader-seconds", type=float, default=10.0)
    ap.add_argument("--skip-leader", action="store_true")
    ap.add_argument("--keep", metavar="DIR", help="keep the check dataset here (new directory)")
    ap.add_argument("--out", default=f"verify/record_check/{time.strftime('%Y%m%d-%H%M%S')}")
    args = ap.parse_args(argv)

    from .piper import PiperBackend, PiperLeader

    cfg = load_config(args.follower_config)
    rep = Report(args.out)
    check_can(rep, list(cfg.arms.values()) + ([] if args.skip_leader else [args.leader]))
    robot = PiperBackend(cfg, passive=True)  # read-only: never enables or commands
    robot.connect()
    action = None
    try:
        time.sleep(0.5)
        check_follower(rep, robot, cfg)
        check_cameras(rep, robot, cfg, args.out, args.fps)
        if not args.skip_leader:
            action = check_leader(rep, PiperLeader(args.leader), cfg, args.leader_seconds)
        free = shutil.disk_usage(".").free / 1e9
        rep.check("dataset", "disk free", free > 20, f"{free:.0f} GB")
        check_dataset(rep, robot, cfg, args.seconds, args.fps, args.keep, action)
    finally:
        robot.close()
        rep.save()
    failed = [s for s in rep.data if rep.failed(s)]
    print(f"\n{'FAILED: ' + ', '.join(failed) if failed else 'ALL PASS'} (report: {rep.path})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
