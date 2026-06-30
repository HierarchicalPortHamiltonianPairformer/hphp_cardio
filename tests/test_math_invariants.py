"""
Tests for mathematical invariants: J skew-symmetry, R positive semi-definiteness, passivity, rho_1.
"""
import torch
import pytest
from models.meso_pairformer import PHPairformerBlock
from models.micro_quantum import QuantumElectrostaticSurrogate

def test_gate_1_J_skew_symmetric():
    model = PHPairformerBlock(d_s=16, d_p=16)
    for _ in range(100):
        S = torch.randn(10, 16)
        P = torch.randn(10, 10, 16)
        J_SS = model.get_J_SS(S)
        assert torch.allclose(J_SS, -J_SS.transpose(-1, -2), atol=1e-6)
        J_PP = model.get_J_PP(P)
        assert torch.allclose(J_PP, -J_PP.transpose(0, 1), atol=1e-6)

def test_gate_2_R_positive_semi_definite():
    model = PHPairformerBlock(d_s=16, d_p=16)
    for i in range(100):
        S = torch.randn(10, 16)
        P = torch.randn(10, 10, 16)
        
        if i == 0:
            L = model.get_L_PP(P)
            R = L @ L.mT
            print('\nDiagnostic:')
            print('L shape:', L.shape)
            print('L lower triangular check:', torch.allclose(L, torch.tril(L)))
            print('R - L@LT norm:', torch.norm(R - L @ L.mT))
            eigvals_diag = torch.linalg.eigvalsh(R)
            print('Min eigenvalue:', eigvals_diag.min().item())

        R_SS = model.get_R_SS(S)
        eigvals_SS = torch.linalg.eigvalsh(R_SS.double())
        assert torch.all(eigvals_SS >= -1e-6)
        
        R_PP = model.get_R_PP(P)
        # float64 required: eigvalsh on float32 LLᵀ introduces 
        # sign ambiguity near zero eigenvalues. float64 resolves 
        # this correctly. R is stored in float32 in production;
        # this cast is test-only for precise gate verification.
        eigvals_PP = torch.linalg.eigvalsh(R_PP.double())
        print(f'Iter {i}: min_eigval = {eigvals_PP.min().item():.2e}')
        assert torch.all(eigvals_PP >= -1e-6)

def test_gate_3_passivity_inequality():
    # passivity_check
    model = PHPairformerBlock(d_s=16, d_p=16)
    dt = 0.01
    for _ in range(100):
        S_k = torch.randn(10, 16)
        P_k = torch.randn(10, 10, 16)
        u = torch.zeros_like(S_k)
        
        H_k = model._compute_H_meso(S_k, P_k).item()
        S_k_plus_1, P_k_plus_1 = model.forward(S_k, P_k, u, dt=dt)
        H_k_plus_1 = model._compute_H_meso(S_k_plus_1, P_k_plus_1).item()
        
        assert H_k_plus_1 <= H_k + 1e-6

def test_gate_4_rho1_properties():
    molecules_N_e = {"dofetilide": 100, "cisapride": 90, "quinidine": 80, "sotalol": 70, "astemizole": 110}
    for mol, N_e in molecules_N_e.items():
        # TEST ONLY: n_qubits=6. Production active space: 10-20 qubits
        # determined by FNO occupation threshold 0.002 < n_p < 1.998
        # Restore to n_qubits=self.n_qubits before any molecular run.
        surrogate = QuantumElectrostaticSurrogate(n_qubits=6, target="qpp-cpu")
        surrogate.build_hamiltonian(n_electrons=N_e)
        surrogate.run_vqe()
        rho_1 = surrogate.get_rho_1()
        assert abs(torch.trace(rho_1).item() - N_e) < 1e-6
        assert torch.norm(rho_1 - rho_1.T.conj(), p='fro') < 1e-6
