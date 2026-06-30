"""
Dirac interfaces for coupling micro-meso (D1) and meso-macro (D2) hierarchies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils.parametrizations as P

class DiracInterface_D1(nn.Module):
    r"""
    Dirac D1 Interface: Micro-to-Meso Coupler.
    Couples Hellmann-Feynman atomic force gradients (micro-scale) to the meso-scale input forces.

    Mathematical Formulation:
    1. Isotropic Projection:
       The input force is projected using an orthogonal parameterization:
       $$u_{\text{meso}} = W_{D1} (-dE_0/dR)$$
       where $W_{D1}$ is constrained to the Stiefel manifold via orthogonal parametrization:
       $$W_{D1}^T W_{D1} = I$$
       This ensures isometry, preserving the norm of the force vector:
       $$\|u_{\text{meso}}\|_2 = \|-dE_0/dR\|_2$$

    2. Power Balance Verification:
       Validates conservation of power across the interface:
       $$\mathcal{P}_{\text{micro}} + \mathcal{P}_{\text{meso}} = e_{\text{micro}}^T f_{\text{micro}} + e_{\text{meso}}^T f_{\text{meso}} = 0$$
       where:
       - $e_{\text{micro}} = dE_0/dR$ (micro effort)
       - $f_{\text{micro}} = dE_0/dR$ (micro flow)
       - $e_{\text{meso}} = u_{\text{meso}}$ (meso effort)
       - $f_{\text{meso}} = -u_{\text{meso}}$ (meso flow)
    """
    def __init__(self, N_atoms, d_s):
        super().__init__()
        self.N_atoms = N_atoms
        self.d_s = d_s
        self.proj = nn.Linear(N_atoms * 3, N_atoms * d_s, bias=False)
        P.orthogonal(self.proj)

    def verify_power_balance(self, e_micro, f_micro, e_meso, f_meso):
        power = (e_micro.flatten() @ f_micro.flatten()) + (e_meso.flatten() @ f_meso.flatten())
        return abs(power.item()) < 1e-5

    def forward(self, dE0_dR, e_micro=None, f_micro=None, e_meso=None, f_meso=None):
        u_meso_flat = self.proj(-dE0_dR.reshape(-1))
        u_meso = u_meso_flat.reshape(self.N_atoms, self.d_s)
        
        if e_micro is not None and f_micro is not None and e_meso is not None and f_meso is not None:
            if not self.verify_power_balance(e_micro, f_micro, e_meso, f_meso):
                print("WARNING: D1 power balance violated!")
                
        return u_meso

class DiracInterface_D2(nn.Module):
    r"""
    Dirac D2 Interface: Meso-to-Macro Coupler.
    Couples the pairformer pair-state representation to the macroscopic IKr drug blockade fraction.

    Mathematical Formulation:
    1. Attention-Pooled Readout:
       Computes the blockade fraction $\theta(t)$ using query-key attention over the pair states $P$:
       $$\theta(t) = \sigma(W_{\text{out}} \cdot \text{AttentionPool}(P) + b_{\text{out}}) \in (0, 1)$$

    2. Power Balance Verification:
       Validates conservation of power across the interface:
       $$\mathcal{P}_{\text{meso}} + \mathcal{P}_{\text{macro}} = e_{\text{meso}}^T f_{\text{meso}} + u_{\text{macro}}^T f_{\text{macro}} = 0$$
       where:
       - $e_{\text{meso}} = \theta(t)$ (meso effort)
       - $f_{\text{meso}} = \theta(t)$ (meso flow)
       - $u_{\text{macro}} = \theta(t)$ (macro input / effort)
       - $f_{\text{macro}} = -u_{\text{macro}}$ (macro flow)
    """
    def __init__(self, d_p, d_attn):
        super().__init__()
        self.d_p = d_p
        self.d_attn = d_attn
        self.W_Q = nn.Linear(d_p, d_attn)
        self.W_K = nn.Linear(d_p, d_attn)
        self.W_out = nn.Linear(d_p, 1)
        
        # Physical prior: assume no blockade before training.
        # sigmoid(-3.0) = 0.047: ~5% IKr blockade at initialisation.
        # This keeps the ORd system in the physiological APD90 range
        # (~271ms) before any drug-specific patterns are learned.
        nn.init.xavier_uniform_(self.W_out.weight, gain=0.1)
        nn.init.constant_(self.W_out.bias, -3.0)

    def verify_power_balance(self, theta_t, f_meso, u_macro, f_macro):
        power = (theta_t * f_meso.sum()) + (u_macro * f_macro)
        return abs(power.item()) < 1e-5

    def forward(self, P, f_meso=None, u_macro=None, f_macro=None):
        Q_pool = self.W_Q(P.mean(dim=1))
        K_pool = self.W_K(P.mean(dim=0))
        
        attn = F.softmax(Q_pool @ K_pool.T / (self.d_attn ** 0.5), dim=-1)
        pooled = (attn.unsqueeze(-1) * P).sum(dim=(0, 1))
        
        theta_t = torch.sigmoid(self.W_out(pooled)).squeeze(-1)
        
        if not (0 < theta_t.item() < 1):
            raise ValueError(f"theta_t out of bounds: {theta_t.item()}")
            
        if f_meso is not None and u_macro is not None and f_macro is not None:
            if not self.verify_power_balance(theta_t, f_meso, u_macro, f_macro):
                print("WARNING: D2 power balance violated!")
                
        return theta_t
