"""
models/ord_cipa_runner.py

Standalone CiPA O'Hara-Rudy 2017 simulation runner via myokit.
Called as a subprocess by ord_evaluator.py.

Usage (direct):
    python3 models/ord_cipa_runner.py \
        --theta  0.05 \
        --ko     5.4 \
        --bcl    1000 \
        --beats  10 \
        --model  data/ord_cipa.cellml

Output (JSON to stdout):
    {"apd90": 269.4, "vmax": 42.1, "vrest": -88.1, "status": "ok"}

Exit codes:
    0 — success
    1 — simulation failure (depolarisation block, NaN, timeout)
    2 — argument error
"""

import argparse
import json
import sys
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="CiPA ORd myokit runner")
    p.add_argument("--theta", type=float, required=True,
                   help="Fractional IKr blockade θ ∈ [0, 1)")
    p.add_argument("--ko",    type=float, default=5.4,
                   help="Extracellular K+ concentration (mmol/L, default 5.4)")
    p.add_argument("--bcl",   type=float, default=1000.0,
                   help="Basic cycle length (ms, default 1000.0)")
    p.add_argument("--beats", type=int,   default=10,
                   help="Number of pacing beats (default 10)")
    p.add_argument("--model", type=str,   default="data/ord_cipa.cellml",
                   help="Path to CiPA ORd CellML file")
    p.add_argument("--rtol",  type=float, default=1e-8)
    p.add_argument("--atol",  type=float, default=1e-10)
    return p.parse_args()


def run_simulation(theta, ko, bcl, n_beats, model_path, rtol, atol):
    import myokit
    import myokit.formats.cellml as cellml

    # Load model
    importer = cellml.CellMLImporter()
    m = importer.model(model_path)

    # Apply θ(t) port: scale GKr by (1 - θ)
    if not (0.0 <= theta < 1.0):
        raise ValueError(f"theta must be in [0, 1), got {theta}")
    gkr_base = float(m.get("IKr.GKr_b").value())
    m.get("IKr.GKr_b").set_rhs(repr(gkr_base * (1.0 - theta)))

    # Apply K_o
    m.get("extracellular.ko").set_rhs(repr(ko))

    # Pacing protocol — MUST bind Istim to pace
    istim = m.get("membrane.Istim")
    istim.set_binding("pace")
    amp = float(m.get("membrane.i_Stim_Amplitude").value())  # -80.0 A/F

    p = myokit.pacing.blocktrain(
        period=bcl,
        duration=0.5,
        offset=0,
        level=amp,
        limit=n_beats,
    )

    s = myokit.Simulation(m, p)
    s.set_tolerance(rtol, atol)

    # Run
    total_time = n_beats * bcl
    d = s.run(total_time, log=["environment.time", "membrane.v"])

    t = np.array(d["environment.time"])
    v = np.array(d["membrane.v"])

    # Sanity checks
    if np.any(np.isnan(v)) or np.any(np.isinf(v)):
        raise RuntimeError("NaN or Inf in voltage trajectory")

    # Extract APD90 from final beat
    beat_start = (n_beats - 1) * bcl
    mask = t >= beat_start
    t10, v10 = t[mask], v[mask]

    if len(t10) < 10:
        raise RuntimeError(f"Too few time points in final beat: {len(t10)}")

    vmax = float(v10.max())
    vrest = float(v10[-1])
    v90 = vmax - 0.9 * (vmax - vrest)

    peak_idx = int(v10.argmax())
    after_t = t10[peak_idx:]
    after_v = v10[peak_idx:]

    cross = np.where(after_v <= v90)[0]
    if len(cross) == 0:
        raise RuntimeError(
            f"Cell did not repolarise to 90%% in final beat "
            f"(Vmax={vmax:.1f}, V90={v90:.1f}, Vend={vrest:.1f})"
        )

    apd90 = float(after_t[cross[0]] - after_t[0])

    # Physiological bounds check
    if not (100.0 < apd90 < 600.0):
        raise RuntimeError(
            f"APD90={apd90:.1f}ms outside physiological range [100, 600] ms"
        )

    return {"apd90": apd90, "vmax": vmax, "vrest": vrest, "status": "ok"}


def main():
    args = parse_args()
    try:
        result = run_simulation(
            theta=args.theta,
            ko=args.ko,
            bcl=args.bcl,
            n_beats=args.beats,
            model_path=args.model,
            rtol=args.rtol,
            atol=args.atol,
        )
        print(json.dumps(result))
        sys.exit(0)
    except Exception as e:
        err = {"apd90": None, "status": "error", "message": str(e)}
        print(json.dumps(err), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
