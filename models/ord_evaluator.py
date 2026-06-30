r"""
Evaluator for O'Hara-Rudy (ORd) CiPA 2017 electrophysiology simulations.
Wraps the standalone subprocess runner or provides a pre-compiled fast simulation loop.
"""

import subprocess
import json
import os
import numpy as np


def evaluate_ord(theta_t: float, K_o: float = 5.4, 
                 BCL: float = 1000.0, n_beats: int = 10) -> float:
    r"""
    Evaluate the APD90 of the port-augmented ORd model via standalone subprocess runner.
    Calling as a subprocess prevents CVODE C extension collisions and memory leakage.

    Mathematical Formulation:
    - Runs the external runner:
      $$python3 models/ord_cipa_runner.py --theta \theta_t --ko K_o --bcl BCL --beats n_beats$$
    - Parses output JSON to retrieve the action potential duration at 90% repolarization (APD90).
    """
    try:
        result = subprocess.run(
            ['python3', 'models/ord_cipa_runner.py',
             '--theta', str(theta_t),
             '--ko', str(K_o),
             '--bcl', str(BCL),
             '--beats', str(n_beats)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(f"myokit runner failed: {result.stderr or result.stdout}")
        
        data = json.loads(result.stdout.strip())
        if data.get("status") == "error":
            raise RuntimeError(data.get("message", "unknown simulation error"))
        return float(data['apd90'])
    except Exception as e:
        raise RuntimeError(f"Simulation failed for theta_t={theta_t}, K_o={K_o}: {e}")


# Baseline (cached)
_baseline_apd90 = None

def get_baseline_apd90(K_o=5.4, BCL=1000.0, cache_path='data/myokit_baseline.json'):
    global _baseline_apd90
    if _baseline_apd90 is not None:
        return _baseline_apd90
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            _baseline_apd90 = json.load(f)['apd90']
        print(f'Loaded myokit baseline APD90: {_baseline_apd90:.2f} ms')
        return _baseline_apd90
    print('Computing myokit baseline APD90...')
    _baseline_apd90 = evaluate_ord(theta_t=0.0, K_o=K_o, BCL=BCL, n_beats=10)
    with open(cache_path, 'w') as f:
        json.dump({'apd90': _baseline_apd90}, f)
    print(f'Cached myokit baseline APD90: {_baseline_apd90:.2f} ms')
    return _baseline_apd90


def predict_delta_qtc(theta_t: float, K_o: float = 5.4, 
                      BCL: float = 1000.0) -> float:
    baseline = get_baseline_apd90()
    apd90 = evaluate_ord(theta_t=theta_t, K_o=K_o, BCL=BCL, n_beats=10)
    return apd90 - baseline


class FastORdEvaluator:
    r"""
    In-process fast evaluator that pre-compiles a single Myokit CVODE simulation.
    Used during training/validation steps to bypass subprocess creation overhead.
    """
    def __init__(self, rtol=1e-8, atol=1e-10):
        import myokit
        import myokit.formats.cellml as cellml
        importer = cellml.CellMLImporter()
        
        # Load a single model and compile a single simulation to prevent CVODE C extension collisions
        self.model = importer.model('data/ord_cipa.cellml')
        istim = self.model.get('membrane.Istim')
        istim.set_binding('pace')
        self.amp = float(self.model.get('membrane.i_Stim_Amplitude').value())
        self.gkr_base = float(self.model.get('IKr.GKr_b').value())
        
        p = myokit.pacing.blocktrain(period=1000.0, duration=0.5, offset=0, level=self.amp, limit=0)
        self.sim = myokit.Simulation(self.model, p)
        self.sim.set_tolerance(rtol, atol)
        
    def evaluate(self, theta_t: float, K_o: float = 5.4, BCL: float = 1000.0, n_beats: int = 10) -> float:
        # If pacing period is 1000ms, use the pre-compiled fast simulation
        if abs(BCL - 1000.0) < 1e-3:
            sim = self.sim
            sim.set_constant('extracellular.ko', float(K_o))
            sim.set_constant('IKr.GKr_b', self.gkr_base * (1.0 - float(theta_t)))
            sim.reset()
            d = sim.run(n_beats * BCL, log=["environment.time", "membrane.v"])
        else:
            # Fallback for OOD stress tests with different BCL (e.g. bradycardia BCL=1500)
            import myokit
            import myokit.formats.cellml as cellml
            importer = cellml.CellMLImporter()
            model_temp = importer.model('data/ord_cipa.cellml')
            istim_temp = model_temp.get('membrane.Istim')
            istim_temp.set_binding('pace')
            p = myokit.pacing.blocktrain(period=float(BCL), duration=0.5, offset=0, level=self.amp, limit=int(n_beats))
            sim = myokit.Simulation(model_temp, p)
            sim.set_tolerance(1e-6, 1e-8)
            sim.set_constant('extracellular.ko', float(K_o))
            sim.set_constant('IKr.GKr_b', self.gkr_base * (1.0 - float(theta_t)))
            d = sim.run(n_beats * BCL, log=["environment.time", "membrane.v"])
        
        t = np.array(d["environment.time"])
        v = np.array(d["membrane.v"])
        
        beat_start = (n_beats - 1) * BCL
        mask = t >= beat_start
        t10, v10 = t[mask], v[mask]
        
        vmax, vrest = v10.max(), v10[-1]
        v90 = vmax - 0.9 * (vmax - vrest)
        pidx = v10.argmax()
        cross = np.where(v10[pidx:] <= v90)[0]
        apd90 = float(t10[pidx + cross[0]] - t10[pidx]) if len(cross) else 999.0
        return apd90
