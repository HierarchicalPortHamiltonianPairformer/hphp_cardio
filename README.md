[![DOI](https://zenodo.org/badge/1266703383.svg)](https://doi.org/10.5281/zenodo.21086450)
# HPHP-Cardio
**A Hierarchical Port-Hamiltonian Pairformer Predicts Drug-Induced Arrhythmia Risk Under Clinical Out-of-Distribution Conditions Without Retraining**

[![Tests](https://img.shields.io/badge/tests-12%20passed-brightgreen)]()
[![Physics](https://img.shields.io/badge/L_passivity-0.00e%2B00-blue)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## Overview


HPHP-Cardio predicts drug-induced QT prolongation and Torsades de Pointes (TdP) risk by bridging three biophysical scales through power-preserving Dirac interfaces:

| Hierarchy | Scale | Method |
|---|---|---|
| I | Quantum molecular | CUDA-Q VQE (UCCSD, active-space FNO) |
| II | Structural binding | Port-Hamiltonian Pairformer |
| III | Cellular electrophysiology | CiPA O'Hara-Rudy 2017 (myokit) |

Physical constraints (energy conservation, thermodynamic passivity) are enforced by algebraic parameterisation — not soft penalties — so they hold exactly at inference, including under out-of-distribution clinical conditions.

**Key results on CiPA 28-compound panel:**
- Calibrated RMSE: **9.77 ms**
- Calibrated MAE vs DeepHERG: **4.58 vs 14.77 ms** (Wilcoxon p = 0.000586)
- AUROC (TdP High vs Low+Med): **0.831** (calibrated), **0.903** (best checkpoint)
- Passivity loss L_passivity: **0.000** across 1,000 training epochs
- Zero-shot hypokalaemia prediction: **21.35 ms** APD₉₀ prolongation (no retraining)

---

## Hardware Requirements

| Component | Minimum | Tested |
|---|---|---|
| GPU | Any CUDA 12.x GPU, 24 GB VRAM | NVIDIA Blackwell GB10, 128 GB unified memory |
| CUDA | 12.0+ | 13.0 |
| CUDA-Q | 0.8+ | 0.9.x |
| RAM | 32 GB | 128 GB (unified) |
| Storage | 10 GB | — |

> **Note:** The quantum hierarchy (Hierarchy I) requires CUDA-Q with a `nvidia` GPU target. On hardware without GB10, reduce `n_qubits` in `models/micro_quantum.py` to ≤ 20 to stay within VRAM limits. All other components run on any CUDA-capable GPU.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/[your-org]/hphp_cardio.git
cd hphp_cardio

# 2. Create a virtual environment (Python 3.12 recommended)
python3 -m venv hphp_cardio_env
source hphp_cardio_env/bin/activate

# 3. Install PyTorch (CUDA 12.4 wheel — adjust for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 4. Install CUDA-Q (Blackwell/CUDA 13 users: see note below)
pip install cuda-quantum --extra-index-url https://pypi.nvidia.com

# 5. Install remaining dependencies
pip install -r requirements.txt

# 6. Verify installation
python3 scripts/verify_environment.py
```

> **CUDA 13 / Blackwell users:** If `pip install cuda-quantum` installs a build that does not support CUDA 13, use the NVIDIA container:
> ```bash
> docker run --gpus all -it --shm-size=120g nvcr.io/nvidia/cuda-quantum:latest
> ```

---

## Project Structure

```
hphp_cardio/
├── data/
│   ├── pipeline.py               # ChEMBL query and CiPA dataset preparation
├── models/
│   ├── micro_quantum.py          # Hierarchy I: CUDA-Q VQE molecular surrogate
│   ├── meso_pairformer.py        # Hierarchy II: Port-Hamiltonian Pairformer
│   ├── dirac_coupler.py          # Dirac interfaces D₁ and D₂
│   ├── ord_cipa_runner.py        # CiPA ORd evaluator via myokit (production)
│   └── ord_evaluator.py          # APD90 prediction interface
├── tests/
│   ├── test_math_invariants.py   # Gates 1–4: J, R, passivity, ρ₁
│   ├── test_port_passivity.py    # Gates 5–7: D₁, D₂, θ bounds
│   └── test_e2e_clinical.py      # Gates 8–12: ORd, OOD, freeze, pharmacology
├── scripts/
│   ├── verify_environment.py     # Installation verification
│   └── calibrate.py              # Post-hoc isotonic regression calibration
├── train.py                      # Two-stage training (ChEMBL + CiPA fine-tuning)
├── evaluate.py                   # Evaluation, ablations, OOD stress tests
├── requirements.txt
├── REPRODUCIBILITY.md            # Step-by-step reproduction guide
└── README.md
```

---

## Quick Start: Reproduce Primary Results

```bash
# Step 1: Run the physics gate test suite (must pass before any training)
pytest tests/ -v --tb=short
# Expected: 12 passed in ~5s (clinical tests use cached myokit evaluator)

# Step 2: Prepare data
python3 data/pipeline.py --chembl --cipa
# Downloads ChEMBL hERG data (~7,400 compounds) and prepares CiPA splits

# Step 3: Stage 1 — ChEMBL pre-training (GFN2-xTB descriptors)
python3 train.py --stage 1 --epochs 50 --device cuda
# Runtime: ~10 min on GB10. Saves: checkpoints/pretrained_h1_h2_xtb.pt

# Step 4: Stage 2 — CiPA fine-tuning (5-fold CV, 200 epochs)
python3 train.py \
    --stage 2 \
    --epochs 200 \
    --folds 5 \
    --pretrained checkpoints/pretrained_h1_h2_xtb.pt \
    --device cuda
# Runtime: ~8 hours on GB10. Saves: checkpoints/stage2_xtb/

# Step 5: Calibrate and evaluate
python3 scripts/calibrate.py --checkpoint checkpoints/stage2_xtb/best_model_fold_5.pt
python3 evaluate.py --checkpoint checkpoints/stage2_xtb/best_model_fold_5.pt --all
```

---

## Inference on a New Compound

```python
from models.meso_pairformer import PHPairformerBlock
from models.dirac_coupler import DiracInterface_D1, DiracInterface_D2
from models.ord_evaluator import predict_delta_qtc
import torch

# Load trained model
checkpoint = torch.load('checkpoints/stage2_xtb/best_model_fold_5.pt')
# ... initialise model and load state_dict ...

# Predict for a new SMILES string
smiles = "CC(C)NCC(O)c1ccc(O)c(O)c1"  # example
delta_qtc_ms, tdp_risk = model.predict(smiles)
print(f"Predicted ΔQTc: {delta_qtc_ms:.1f} ms")
print(f"TdP risk class: {tdp_risk}")  # High / Medium / Low
```

---

## Running the Physics Gate Suite

All 12 gates must pass before any gradient computation. They verify mathematical invariants that the architecture guarantees by construction.

```bash
pytest tests/ -v
```

```
tests/test_math_invariants.py::test_gate_1_J_skew_symmetric          PASSED
tests/test_math_invariants.py::test_gate_2_R_positive_semi_definite  PASSED
tests/test_math_invariants.py::test_gate_3_passivity_inequality       PASSED
tests/test_math_invariants.py::test_gate_4_rho1_properties            PASSED
tests/test_port_passivity.py::test_gate_5_D1_power_balance            PASSED
tests/test_port_passivity.py::test_gate_6_D2_power_balance            PASSED
tests/test_port_passivity.py::test_gate_7_theta_bounds                PASSED
tests/test_e2e_clinical.py::test_gate_8_hypokalemia                   PASSED
tests/test_e2e_clinical.py::test_gate_9_bradycardia                   PASSED
tests/test_e2e_clinical.py::test_gate_10_freeze                       PASSED
tests/test_e2e_clinical.py::test_gate_11_e2e_forward                  PASSED
tests/test_e2e_clinical.py::test_gate_12_pharmacological_ordering     PASSED

12 passed in 4.92s
```

---

## Requirements

```
# requirements.txt — see file for full pinned versions
torch>=2.3.0
cuda-quantum>=0.8.0
torchdiffeq>=0.2.3
pyscf>=2.5.0
openfermion>=1.6.0
openfermionpyscf>=0.5
rdkit>=2023.09.1
scipy>=1.12.0
scikit-learn>=1.4.0
myokit>=1.39.2
xtb-python>=22.1
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
wandb>=0.16.0
pytest>=9.0.0
```

---

## Citation

```bibtex
@article{sellapandian2026hphp,
  title   = {A Hierarchical Port-Hamiltonian Pairformer Predicts Drug-Induced Arrhythmia Risk Under Clinical Out-of-Distribution Conditions Without Retraining},
  author  = {Sellapandian, Sonale},
  journal = {PLOS Computational Biology},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT License. See `LICENSE` for details.

The CiPA O'Hara-Rudy 2017 model is used via myokit under the BSD licence of the original CiPA publication (Li et al. 2017). The model file `data/ord_cipa.cellml` is sourced from the Physiome Model Repository.

