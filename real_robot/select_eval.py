"""Selectors over pi0.5 candidates: medoid (consensus), JEV-lite phase-gated, verifier, combos."""
import json, os, sys
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from real_robot.jevlite import JevLite
from real_robot.verifier import dist, chunk_feat, load

z = np.load("outputs/verifier/samples_004033.npz", allow_pickle=True)
meta = json.loads(str(z["meta"])); C, demo, state = z["cands"], z["demo"], z["state"]
Fn, I, Sn, N = C.shape[:4]
scales = meta["scales"]
ds = LeRobotDataset(repo_id="cube_grid_all_pi05", root="demos/cube_grid_all_pi05")
lite = JevLite("outputs/jevlite/model_dinos_g4_aug")
ver = load("outputs/verifier/v2")
P = []
X = []
for it in meta["items"]:
    x = ds[it["idx"]]
    im = {c: (x[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for c in ("wrist", "base")}
    r = lite({"cam_wrist": im["wrist"], "cam_high": im["base"]})
    P.append(r)
    X.append(np.concatenate([lite.enc([im["wrist"]]).reshape(-1), lite.enc([im["base"]]).reshape(-1)]))
print("lite keys", list(P[0].keys())[:10])
X = np.stack(X).astype(np.float32)


def phase_p(r, names):
    ph = r.get("phases", r)
    return float(sum(ph.get(n, 0.0) for n in names))


def run(sel_I, sel_S, n, method):
    out_d, out_close = [], []
    for k in range(Fn):
        c = C[k][sel_I][:, sel_S][..., :n, :, :].reshape(-1, C.shape[-2], 7)
        d = dist(c, demo[k][None])
        g0 = state[k, 6]
        closes = c[..., 6].min(-1) < 0.6 * g0
        pool = np.arange(len(c))
        if "gate" in method:
            pg = phase_p(P[k], ["grasp", "holding"]) if g0 > 0.5 else 0.0
            want = pg > 0.5
            m = closes == want
            if m.any():
                pool = pool[m]
        if "medoid" in method:
            cc = c[pool]
            D = dist(cc[:, None], cc[None])
            b = pool[D.mean(1).argmin()]
        elif "verifier" in method:
            cf = chunk_feat(c[pool], state[k][None].repeat(len(pool), 0)).astype(np.float32)
            with torch.no_grad():
                s = ver(torch.tensor(X[k:k+1], device="cuda"), torch.tensor(state[k:k+1], device="cuda"),
                        torch.tensor(cf[None], device="cuda"))[0].cpu().numpy()
            b = pool[s.argmax()]
        else:
            b = pool[0]
        out_d.append(d[b]); out_close.append(closes[b])
    dc = demo[..., 6].min(-1) < 0.6 * state[:, 6]
    od, oc = np.array(out_d), np.array(out_close)
    return {"dist": round(float(od.mean()), 4), "close_when_demo_closes": round(float(oc[dc].mean()), 3),
            "close_when_demo_doesnt": round(float(oc[~dc].mean()), 3)}

allI = np.ones(I, bool); main = np.zeros(I, bool); main[0] = True
rep = {}
for sname, ss in {"s0.5": [0], "s0.5+1": [0, 1], "all": [0, 1, 2, 3]}.items():
    sm = np.zeros(Sn, bool); sm[ss] = True
    for iname, im_ in {"main": main, "allinstr": allI}.items():
        for m in ("single", "medoid", "verifier", "gate_medoid", "gate_verifier"):
            n = 1 if m == "single" else N
            if m == "single" and (iname != "main" or sname != "s0.5+1"):
                continue
            if m == "single":
                sm1 = np.zeros(Sn, bool); sm1[1] = True
                rep["single(s1)"] = run(main, sm1, 1, m); continue
            rep[f"{m}|{iname}|{sname}"] = run(im_, sm, n, m)
for k, v in rep.items():
    print(k, v)
json.dump(rep, open("outputs/verifier/select_eval.json", "w"), indent=1)
