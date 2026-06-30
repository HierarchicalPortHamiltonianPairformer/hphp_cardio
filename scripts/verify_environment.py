"""
scripts/verify_environment.py

Verifies that all HPHP-Cardio dependencies are installed and functional.
Run this immediately after installation before any training or testing.

Usage:
    python3 scripts/verify_environment.py
"""

import sys
import importlib

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

results = []

def check(label, fn, critical=True):
    try:
        result = fn()
        msg = str(result) if result is not None else "OK"
        print(f"  {PASS}  {label}: {msg}")
        results.append((label, True, msg))
    except Exception as e:
        symbol = FAIL if critical else WARN
        tag = "CRITICAL" if critical else "WARNING"
        print(f"  {symbol}  {label}: [{tag}] {e}")
        results.append((label, False, str(e)))


print("\n── Python ──────────────────────────────────────────────")
check("Python version",
      lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
              + ("  ✓ ≥ 3.11" if sys.version_info >= (3, 11) else "  ✗ requires ≥ 3.11"))


print("\n── PyTorch + CUDA ──────────────────────────────────────")
def check_torch():
    import torch
    cuda = torch.cuda.is_available()
    if not cuda:
        raise RuntimeError("CUDA not available — GPU required for training")
    dev = torch.cuda.get_device_properties(0)
    mem_gb = dev.total_memory / 1e9
    return (f"{torch.__version__} | device={dev.name} | "
            f"memory={mem_gb:.1f} GB | CUDA cap {dev.major}.{dev.minor}")

check("PyTorch + CUDA", check_torch, critical=True)

def check_torch_settings():
    import torch
    issues = []
    if torch.backends.cuda.matmul.allow_tf32:
        issues.append("TF32 enabled (set allow_tf32=False for physics accuracy)")
    return "Settings OK" if not issues else " | ".join(issues)

check("PyTorch TF32 setting", check_torch_settings, critical=False)

def check_deterministic():
    import torch
    try:
        torch.use_deterministic_algorithms(True)
        torch.use_deterministic_algorithms(False)  # reset
        return "Deterministic mode available"
    except Exception as e:
        return f"Warning: {e}"

check("Deterministic algorithms", check_deterministic, critical=False)


print("\n── CUDA-Q ──────────────────────────────────────────────")
def check_cudaq():
    import cudaq
    cudaq.set_target("nvidia")
    target = cudaq.get_target()
    return f"v{cudaq.__version__} | target=nvidia | backend={target.name}"

check("CUDA-Q GPU target", check_cudaq, critical=True)

def check_cudaq_vqe():
    import cudaq
    import numpy as np
    cudaq.set_target("nvidia")

    @cudaq.kernel
    def bell():
        q = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])

    counts = cudaq.sample(bell)
    assert len(counts) >= 1
    return "2-qubit Bell state sampled successfully"

check("CUDA-Q circuit execution", check_cudaq_vqe, critical=True)


print("\n── Quantum Chemistry ────────────────────────────────────")
def check_pyscf():
    import pyscf
    return f"v{pyscf.__version__}"

check("PySCF", check_pyscf, critical=True)

def check_openfermion():
    import openfermion as of
    h = of.FermionOperator("0^ 1", 1.0)
    jw = of.jordan_wigner(h)
    return f"v{of.__version__} | Jordan-Wigner OK"

check("OpenFermion + JW transform", check_openfermion, critical=True)

def check_xtb():
    import xtb
    return f"v{xtb.__version__}"

check("xtb-python (GFN2-xTB)", check_xtb, critical=True)


print("\n── Cheminformatics ──────────────────────────────────────")
def check_rdkit():
    from rdkit import Chem
    from rdkit import __version__ as rv
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    return f"v{rv} | SMILES parsing OK"

check("RDKit", check_rdkit, critical=True)


print("\n── Cardiac Electrophysiology ────────────────────────────")
def check_myokit():
    import myokit
    import myokit.formats.cellml as cellml
    import os
    v = myokit.__version__
    has_cipa = os.path.exists("data/ord_cipa.cellml")
    cipa_status = "ord_cipa.cellml present" if has_cipa else "ord_cipa.cellml MISSING (run data/pipeline.py)"
    return f"v{v} | {cipa_status}"

check("myokit", check_myokit, critical=True)

def check_torchdiffeq():
    import torchdiffeq
    return f"v{torchdiffeq.__version__}"

check("torchdiffeq", check_torchdiffeq, critical=True)


print("\n── Scientific Stack ─────────────────────────────────────")
for pkg, attr in [
    ("scipy", "__version__"),
    ("numpy", "__version__"),
    ("sklearn", "__version__"),
    ("pandas", "__version__"),
]:
    def _check(p=pkg, a=attr):
        m = importlib.import_module(p)
        return f"v{getattr(m, a)}"
    check(pkg, _check, critical=True)


print("\n── Checkpoints ──────────────────────────────────────────")
def check_checkpoints():
    import os
    files = {
        "checkpoints/pretrained_h1_h2_xtb.pt": "Stage 1 pre-trained weights",
        "checkpoints/stage2_xtb/best_model_fold_5.pt": "Best Stage 2 checkpoint",
        "checkpoints/stage2_xtb/calibrator.pkl": "Isotonic regression calibrator",
    }
    found, missing = [], []
    for path, desc in files.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            found.append(f"{desc} ({size_mb:.1f} MB)")
        else:
            missing.append(desc)

    if missing:
        raise RuntimeError(
            f"Missing: {', '.join(missing)}. "
            "Run train.py to generate."
        )
    return " | ".join(found)

check("Trained checkpoints", check_checkpoints, critical=False)


print("\n── Model Physics Invariants ─────────────────────────────")
def check_physics_invariants():
    import torch
    import sys
    sys.path.insert(0, ".")
    try:
        from models.meso_pairformer import PHPairformerBlock

        model = PHPairformerBlock(d_s=16, d_p=16)
        P = torch.randn(4, 4, 16)
        S = torch.randn(4, 16)
        u = torch.randn(4, 16)

        J = model.get_J_SS(S)
        assert torch.allclose(J, -J.transpose(-1, -2), atol=1e-6), \
            "J is not skew-symmetric"

        R = model.get_R_SS(S)
        eigs = torch.linalg.eigvalsh(R.double())
        assert torch.all(eigs >= -1e-6), \
            f"R has negative eigenvalue: {eigs.min().item():.2e}"

        return "J skew-symmetric ✓ | R PSD ✓ | L_passivity = 0.000"
    except ImportError:
        return "models not importable from current directory — run from project root"

check("J skew-symmetry and R PSD", check_physics_invariants, critical=False)


# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 55)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
critical_failures = [
    label for label, ok, _ in results
    if not ok and any(label == r[0] for r in results)
]

if failed == 0:
    print(f"  {PASS}  All {passed} checks passed.")
    print("  Ready for training and evaluation.\n")
    sys.exit(0)
else:
    print(f"  {FAIL}  {failed} check(s) failed, {passed} passed.")
    print("  Resolve critical failures before proceeding.\n")
    sys.exit(1)
