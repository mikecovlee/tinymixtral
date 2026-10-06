#!/usr/bin/env python
"""Measure routing expressiveness: CPT router vs the trained v3.0 linear router.

Two questions are answered separately.

(1) STRUCTURAL bound (model-free, exact).
    v3.0:   pi = softmax(W x),  W in R^{4x1024}, no bias.  As x ranges over
            R^1024 the logits range over all of R^4, so pi ranges over the whole
            open 3-simplex.  Sharpness is unbounded: it grows with ||x||.
    CPT:    pi = B^T q,  q in Delta^7,  B in R^{8x4} fixed per commit.
            pi is therefore confined to conv(rows of B), a polytope with at most
            8 vertices, INDEPENDENT of ||x||.  Sharpness is capped by `energy`.

    We report the L1 radius of that polytope from the simplex centre
    u = (1/4,1/4,1/4,1/4).  Full range is [0, 1.5]; 1.5 means a vertex
    (one-hot) is reachable, 0 means pi is pinned at uniform.

(2) EMPIRICAL distribution on real tokens.
    Each router is run on its own model's hidden states over a real val batch.
    We report the renormalised top-2 weight that actually enters the FFN mix,
    which is what routing expressiveness buys in practice.

Caveat printed with the results: the two checkpoints have very different
training budgets (v3.0 s4_8b = 8.05B tokens / 16 layers; CPT proxy = 100M
tokens / 8 layers), so (2) is not a like-for-like quality comparison.  (1) is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

U = torch.tensor([0.25, 0.25, 0.25, 0.25])


def l1_from_uniform(pi: torch.Tensor) -> torch.Tensor:
    """pi: [..., 4] -> [...], L1 distance from the simplex centre."""
    return (pi - U.to(pi.device, pi.dtype)).abs().sum(dim=-1)


def top2_stats(pi: torch.Tensor) -> dict:
    """Renormalised top-2 weight distribution.  w_max in [0.5, 1.0]."""
    top2 = torch.topk(pi.float(), 2, dim=-1).values
    w = top2 / top2.sum(dim=-1, keepdim=True)
    wmax = w[..., 0]
    ent = -(pi.float().clamp_min(1e-12).log() * pi.float()).sum(-1)
    return {
        "w_max_mean": wmax.mean().item(),
        "w_max_p05": wmax.quantile(0.05).item(),
        "w_max_p50": wmax.quantile(0.50).item(),
        "w_max_p95": wmax.quantile(0.95).item(),
        "frac_w_gt_0.60": (wmax > 0.60).float().mean().item(),
        "frac_w_gt_0.80": (wmax > 0.80).float().mean().item(),
        "pi_entropy_mean": ent.mean().item(),
        "pi_l1_from_uniform_mean": l1_from_uniform(pi).mean().item(),
        "pi_l1_from_uniform_max": l1_from_uniform(pi).max().item(),
    }


# --------------------------------------------------------------------------- #
# CPT structural bound
# --------------------------------------------------------------------------- #
def cpt_B(energy: torch.Tensor, price: torch.Tensor, tau_e: float) -> torch.Tensor:
    """Replicates router.py:447-450 exactly."""
    centered = energy - energy.mean(dim=-1, keepdim=True)
    logits = (centered - price.detach().unsqueeze(0)) / tau_e
    return F.softmax(logits, dim=-1, dtype=torch.float32)


def structural_cpt(ckpt_dir: Path, tau_e: float) -> None:
    sd = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("layers.")})
    print(f"\n=== STRUCTURAL: CPT  ({ckpt_dir.parent.name}, {len(layers)} layers) ===")
    print(f"{'lyr':>3} {'||P||_F':>8} {'|energy|max':>11} {'B radius':>9} "
          f"{'B min':>7} {'B max':>7} {'|price|max':>10}")
    radii = []
    for li in layers:
        p = f"layers.{li}.moe.cpt_router."
        P, energy, price = sd[p + "projection"], sd[p + "energy"], sd[p + "congestion_price"]
        B = cpt_B(energy, price, tau_e)
        radius = l1_from_uniform(B).max().item()
        radii.append(radius)
        print(f"{li:>3} {P.norm().item():8.4f} {energy.abs().max().item():11.4f} "
              f"{radius:9.4f} {B.min().item():7.4f} {B.max().item():7.4f} "
              f"{price.abs().max().item():10.4f}")
    print(f"  -> conv(rows B) L1 radius: mean {sum(radii)/len(radii):.4f}, "
          f"min {min(radii):.4f}, max {max(radii):.4f}   [full simplex = 1.5]")


def structural_v3(ckpt_dir: Path, label: str = "v3.0 linear") -> None:
    sd = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("layers.")})
    print(f"\n=== STRUCTURAL: {label}  ({ckpt_dir}, {len(layers)} layers) ===")
    print(f"{'lyr':>3} {'||W||_2':>9} {'||W||_F':>9} {'|W|max':>8}   "
          f"(logit scale = ||W||_2 * ||x||, unbounded in ||x||)")
    for li in layers:
        W = sd[f"layers.{li}.moe.router.weight"].float()
        s = torch.linalg.svdvals(W)
        print(f"{li:>3} {s[0].item():9.4f} {W.norm().item():9.4f} {W.abs().max().item():8.4f}")
    print("  -> reachable set = entire open 3-simplex; radius 1.5 attainable "
          "for large enough ||x||")


# --------------------------------------------------------------------------- #
# Empirical: run each model on real tokens, hook every router
# --------------------------------------------------------------------------- #
def real_batch(val_pt: Path, bs: int, seq: int, device) -> torch.Tensor:
    t = torch.load(val_pt, map_location="cpu", weights_only=True)
    if isinstance(t, dict):
        t = next(v for v in t.values() if torch.is_tensor(v))
    need = bs * (seq + 1)
    flat = t.reshape(-1)[: need + 1024]
    return flat[: bs * (seq + 1)].view(bs, seq + 1).to(device)


def empirical_v3(repo: Path, ckpt_dir: Path, batch: torch.Tensor) -> dict:
    sys.path.insert(0, str(repo))
    from model import TinyMixtralConfig, TinyMixtralForCausalLM  # noqa: E402

    cfg = TinyMixtralConfig(**json.loads((ckpt_dir / "config.json").read_text()))
    model = TinyMixtralForCausalLM(cfg)
    sd = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model = model.to("cuda").to(torch.bfloat16).eval()

    pis, xnorms = {}, {}

    def hook(li):
        def fn(_mod, inp, out):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            xnorms.setdefault(li, []).append(x.norm(dim=-1).mean().item())
            pis.setdefault(li, []).append(F.softmax(out.reshape(-1, out.shape[-1]).float(), dim=-1).cpu())
        return fn

    hs = [model.layers[i].moe.router.register_forward_hook(hook(i)) for i in range(cfg.num_hidden_layers)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(batch[:, :-1], labels=batch[:, 1:])
    for h in hs:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return {"pi": {k: torch.cat(v) for k, v in pis.items()},
            "xnorm": {k: sum(v) / len(v) for k, v in xnorms.items()}}


def empirical_cpt(repo: Path, cfg_path: Path, ckpt_dir: Path, batch: torch.Tensor) -> dict:
    sys.path.insert(0, str(repo))
    from cpt_model import config_from_json_file, model_for_config  # noqa: E402

    cfg = config_from_json_file(cfg_path)
    model = model_for_config(cfg)
    sd = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model = model.to("cuda").to(torch.bfloat16).eval()

    pis, xnorms = {}, {}

    def hook(li):
        def fn(mod, inp, out):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            xnorms.setdefault(li, []).append(x.norm(dim=-1).mean().item())
            pis.setdefault(li, []).append(out.probabilities.reshape(-1, out.probabilities.shape[-1]).float().cpu())
        return fn

    mods = [model.layers[i].moe.cpt_router for i in range(cfg.num_hidden_layers)]
    hs = [m.register_forward_hook(hook(i)) for i, m in enumerate(mods)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(batch[:, :-1], labels=batch[:, 1:])
    for h in hs:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return {"pi": {k: torch.cat(v) for k, v in pis.items()},
            "xnorm": {k: sum(v) / len(v) for k, v in xnorms.items()}}


def report(tag: str, res: dict) -> None:
    print(f"\n=== EMPIRICAL: {tag} ===")
    hdr = (f"{'lyr':>3} {'||x||':>7} {'w_max':>6} {'p05':>6} {'p50':>6} {'p95':>6} "
           f"{'>0.60':>6} {'>0.80':>6} {'H(pi)':>6} {'L1u':>6} {'L1u_max':>7}")
    print(hdr)
    agg = {k: [] for k in ("w_max_mean", "frac_w_gt_0.60", "frac_w_gt_0.80",
                           "pi_entropy_mean", "pi_l1_from_uniform_mean")}
    for li in sorted(res["pi"]):
        s = top2_stats(res["pi"][li])
        for k in agg:
            agg[k].append(s[k])
        print(f"{li:>3} {res['xnorm'][li]:7.2f} {s['w_max_mean']:6.3f} {s['w_max_p05']:6.3f} "
              f"{s['w_max_p50']:6.3f} {s['w_max_p95']:6.3f} {s['frac_w_gt_0.60']:6.3f} "
              f"{s['frac_w_gt_0.80']:6.3f} {s['pi_entropy_mean']:6.3f} "
              f"{s['pi_l1_from_uniform_mean']:6.3f} {s['pi_l1_from_uniform_max']:7.3f}")
    m = {k: sum(v) / len(v) for k, v in agg.items()}
    print(f"  MEAN over layers: w_max {m['w_max_mean']:.3f}  "
          f"frac>0.60 {m['frac_w_gt_0.60']:.3f}  frac>0.80 {m['frac_w_gt_0.80']:.3f}  "
          f"H(pi) {m['pi_entropy_mean']:.3f} (uniform=1.386)  L1u {m['pi_l1_from_uniform_mean']:.3f}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-repo", default="/home/mikecovlee/work/tinymixtral")
    ap.add_argument("--v3-ckpt", default="/home/mikecovlee/work/tinymixtral/checkpoints/s4_8b/step_0038811_final")
    ap.add_argument("--cpt-repo", default="/home/mikecovlee/work/tinymixtral-improve/git-cpt-v3.0")
    ap.add_argument("--cpt-runs", default="base,price_strong,T_sharp")
    ap.add_argument("--val-pt", default="/mnt/nas/jetpack-hdd/tinymixtral-improve/"
                                        "v3.0-dense-active/data/pretrain/pilot_blend30_val/val_0000.pt")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--sweep-dir", default="checkpoints/hp_sweep")
    ap.add_argument("--linear-proxy-ckpt", default=None,
                    help="checkpoint of the MATCHED-BUDGET linear-router proxy "
                         "(scripts/run_linear_proxy.sh). Defaults to "
                         "<cpt-repo>/checkpoints/linear_proxy/linear/step_0002035_final "
                         "when that directory exists. This is the like-for-like control "
                         "for the CPT proxy runs: same 8L backbone, same 100M tokens, "
                         "same seed/data/schedule; only the router differs.")
    args = ap.parse_args()

    if args.linear_proxy_ckpt is None:
        cand = Path(args.cpt_repo) / "checkpoints/linear_proxy/linear/step_0002035_final"
        args.linear_proxy_ckpt = str(cand) if cand.exists() else ""
    linear_proxy = Path(args.linear_proxy_ckpt) if args.linear_proxy_ckpt else None
    if linear_proxy is not None and not (linear_proxy / "pytorch_model.bin").exists():
        print(f"[warn] linear proxy ckpt not found at {linear_proxy}; "
              f"run scripts/run_linear_proxy.sh first")
        linear_proxy = None

    # ---- (1) structural, model-free ----
    structural_v3(Path(args.v3_ckpt))
    if linear_proxy is not None:
        structural_v3(linear_proxy, label="linear proxy (MATCHED BUDGET, 8L)")
    cpt_cfg = json.loads(Path(args.cpt_repo, "configs/cpt_hp_sweep/proxy_base_8l.json").read_text())
    taus = {"base": 0.6, "price_strong": 0.6, "T_sharp": 0.3, "T_smooth": 1.2}
    for run in args.cpt_runs.split(","):
        run = run.strip()
        tau = taus.get(run, cpt_cfg.get("cpt_expert_temperature", 0.6))
        tau = float(tau) if not isinstance(tau, str) else eval(tau)  # noqa: S307
        ck = Path(args.cpt_repo) / args.sweep_dir / run / "step_0002035_final"
        if ck.exists():
            structural_cpt(ck, tau)

    # ---- (2) empirical on real tokens ----
    device = "cuda"
    batch = real_batch(Path(args.val_pt), args.bs, args.seq, device)
    print(f"\nreal batch: {tuple(batch.shape)}  tokens {batch.min().item()}..{batch.max().item()}")

    res_v3 = empirical_v3(Path(args.v3_repo), Path(args.v3_ckpt), batch)
    m_v3 = report("v3.0 s4_8b (8.05B tok, 16L) linear router", res_v3)

    m_lin = None
    if linear_proxy is not None:
        res_lin = empirical_v3(Path(args.cpt_repo), linear_proxy, batch)
        m_lin = report("linear proxy (100M tok, 8L) — MATCHED BUDGET control", res_lin)

    ms = {}
    for run in args.cpt_runs.split(","):
        run = run.strip()
        ck = Path(args.cpt_repo) / args.sweep_dir / run / "step_0002035_final"
        cfgp = Path(args.cpt_repo) / "configs/cpt_hp_sweep" / f"proxy_{run}_8l.json"
        if not ck.exists():
            continue
        res = empirical_cpt(Path(args.cpt_repo), cfgp, ck, batch)
        ms[run] = report(f"CPT {run} (100M tok, 8L)", res)

    print("\n" + "=" * 78)
    print("SUMMARY  (renormalised top-2 weight w_max; 0.5 = no preference, 1.0 = decisive)")
    print(f"{'model':34s} {'w_max':>7} {'frac>0.6':>9} {'frac>0.8':>9} {'H(pi)':>7}")
    print(f"{'v3.0 linear (8.05B tok, 16L)':34s} {m_v3['w_max_mean']:7.3f} "
          f"{m_v3['frac_w_gt_0.60']:9.3f} {m_v3['frac_w_gt_0.80']:9.3f} {m_v3['pi_entropy_mean']:7.3f}")
    if m_lin is not None:
        print(f"{'linear proxy (100M tok, 8L)':34s} {m_lin['w_max_mean']:7.3f} "
              f"{m_lin['frac_w_gt_0.60']:9.3f} {m_lin['frac_w_gt_0.80']:9.3f} "
              f"{m_lin['pi_entropy_mean']:7.3f}")
    for run, m in ms.items():
        print(f"{'CPT ' + run + ' (100M tok, 8L)':34s} {m['w_max_mean']:7.3f} "
              f"{m['frac_w_gt_0.60']:9.3f} {m['frac_w_gt_0.80']:9.3f} {m['pi_entropy_mean']:7.3f}")

    if m_lin is not None:
        print("\nMATCHED-BUDGET VERDICT (linear proxy vs CPT proxy: same 8L backbone,")
        print("same 100M tokens, same seed/data/schedule — only the router differs):")
        ref = ms[next(iter(ms))] if ms else None
        ref_name = next(iter(ms)) if ms else None
        for key, name in (("w_max_mean", "w_max"),
                          ("frac_w_gt_0.80", "frac>0.8"),
                          ("pi_entropy_mean", "H(pi)"),
                          ("pi_l1_from_uniform_mean", "L1u")):
            if ref is None:
                print(f"  {name:10s} linear {m_lin[key]:7.3f}")
            else:
                print(f"  {name:10s} linear {m_lin[key]:7.3f}   CPT {ref_name} "
                      f"{ref[key]:7.3f}   gap {m_lin[key] - ref[key]:+7.3f}")
        print("  A large gap here is STRUCTURAL (budget is controlled for).")
        print("  A small gap means the v3.0-vs-CPT gap in the table above is a budget artifact.")

    print("\nCAVEAT: the v3.0 row's training budget differs 65x from the proxy rows")
    print("(8.05B vs 100M tokens) and depth 2x (16L vs 8L), so v3.0-vs-CPT is NOT a")
    print("like-for-like quality comparison; it shows each router's operating regime.")
    print("The linear-proxy row above IS like-for-like with the CPT rows.  The")
    print("structural block is exact and budget-independent.")


if __name__ == "__main__":
    main()
