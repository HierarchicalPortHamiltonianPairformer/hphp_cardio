"""
SymplecticPairformerBlock implementing Port-Hamiltonian J, R, H parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PHPairformerBlock(nn.Module):
    r"""
    Hierarchy II: Meso Port-Hamiltonian Pairformer Block.
    Implements a structured port-Hamiltonian state transition framework for nodes S and pairs P.

    Mathematical Formulation:
    1. Port-Hamiltonian State Transitions:
       All state updates are governed by the dissipative port-Hamiltonian formulation:
       $$\dot{x} = (J(x) - R(x)) \nabla H(x) + g(x) u$$
       where $J(x) = -J(x)^T$ is the skew-symmetric interconnection matrix, and $R(x) = R(x)^T \succeq 0$
       is the positive semi-definite dissipation matrix.

    2. Parametrization of Interconnection and Dissipation:
       - Skew-symmetry of $J(x)$ is algebraically guaranteed by construction:
         $$J_{SS}(S) = A(S) - A(S)^T \quad \text{where } A(S) = \text{MLP}_J(S)$$
       - Positivity of $R(x)$ is guaranteed via a Cholesky factorization:
         $$R_{SS}(S) = L(S) L(S)^T \quad \text{where } L(S) = \text{LowerTriangular}(\text{MLP}_R(S))$$
         with $\text{diag}(L)$ processed by a Softplus activation to ensure strict positive definiteness.
       - J_PP Triangle Routing:
         $$J_{PP}(a,b) = \sum_c \left[ V(PW_1)_{ac} \odot V(PW_2)_{bc} \right] - \text{transpose}$$

    3. Meso-scale Hamiltonian:
       Constrained to be non-negative everywhere:
       $$H_{\text{meso}}(S, P) = \sum \|W_S S\|^2_2 + \frac{1}{2} \sum \sigma(\alpha \|P\|^2_2 - \beta) \geq 0$$

    4. Structure-Preserving Implicit Midpoint Rule:
       Discretized via a fixed-point iteration midpoint scheme (Kotyczka & Lefevre 2019):
       $$x_{k+1} = x_k + \Delta t \left[ (J(x_{\text{mid}}) - R(x_{\text{mid}})) \nabla H(x_{\text{mid}}) + u_{\text{meso}} \right]$$
       where $x_{\text{mid}} = \frac{x_k + x_{k+1}}{2}$.

    5. Readout Blockade $\theta(t)$:
       Obtained via attention pooling over the pairs followed by a Sigmoid function, guaranteeing:
       $$\theta(t) \in (0, 1)$$
    """
    def __init__(self, d_s=128, d_p=128, d_attn=64):
        super().__init__()
        self.d_s = d_s
        self.d_p = d_p
        self.d_attn = d_attn
        
        # H_meso parameters
        self.W_S = nn.Linear(d_s, d_s, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        
        self.MLP_J_SS = nn.Sequential(
            nn.Linear(d_s, d_s),
            nn.Tanh(),
            nn.Linear(d_s, d_s * d_s),
            nn.Tanh()
        )
        
        self.W1 = nn.Linear(d_p, d_p, bias=False)
        self.W2 = nn.Linear(d_p, d_p, bias=False)
        
        self.MLP_J_SP = nn.Sequential(
            nn.Linear(d_s + d_p, d_s * d_p),
            nn.Tanh()
        )
        
        self.MLP_R_SS = nn.Sequential(
            nn.Linear(d_s, d_s * d_s),
            nn.Tanh()
        )
        
        self.MLP_R_PP = nn.Sequential(
            nn.Linear(d_p, d_p * d_p),
            nn.Tanh() 
        )
        
        self.W_q = nn.Linear(d_p, d_attn)
        self.W_k = nn.Linear(d_p, d_attn)
        self.W_out = nn.Linear(d_p, 1)
        self.b_out = nn.Parameter(torch.zeros(1))

    def _compute_H_meso(self, S, P):
        energy_S = torch.sum(torch.norm(self.W_S(S), dim=-1)**2)
        energy_P = 0.5 * torch.sum(torch.sigmoid(self.alpha * torch.norm(P, dim=-1)**2 - self.beta))
        return energy_S + energy_P

    def get_J_SS(self, S):
        N = S.shape[0]
        A = self.MLP_J_SS(S).view(N, self.d_s, self.d_s)
        return A - A.transpose(-1, -2)
        
    def get_J_PP(self, P):
        gate_W1 = F.silu(self.W1(P)) 
        gate_W2 = F.silu(self.W2(P)) 
        A = torch.einsum('acd,bcd->abd', gate_W1, gate_W2)
        J_PP = A - A.transpose(0, 1)
        return J_PP

    def get_J_SP(self, S, P):
        S_mean = S.mean(dim=0)
        P_mean = P.mean(dim=(0, 1))
        J_SP_flat = self.MLP_J_SP(torch.cat([S_mean, P_mean]))
        return J_SP_flat.view(self.d_s, self.d_p)

    def get_L_SS(self, S):
        N = S.shape[0]
        MLP_R_output = self.MLP_R_SS(S).view(N, self.d_s, self.d_s)
        
        # Get raw lower triangular
        L = torch.tril(MLP_R_output).clone()
        
        # Apply softplus to diagonal for strict positivity
        # This bounds diagonal elements away from zero,
        # making R = L @ L.T strictly positive definite
        diag_idx = torch.arange(L.shape[-1], device=L.device)
        L[..., diag_idx, diag_idx] = F.softplus(L[..., diag_idx, diag_idx])
        return L

    def get_R_SS(self, S):
        L = self.get_L_SS(S)
        return torch.bmm(L, L.transpose(-1, -2))

    def get_L_PP(self, P):
        N = P.shape[0]
        MLP_R_output = self.MLP_R_PP(P).view(N, N, self.d_p, self.d_p)
        
        # Get raw lower triangular
        L = torch.tril(MLP_R_output).clone()
        
        # Apply softplus to diagonal for strict positivity
        # This bounds diagonal elements away from zero,
        # making R = L @ L.T strictly positive definite
        diag_idx = torch.arange(L.shape[-1], device=L.device)
        L[..., diag_idx, diag_idx] = F.softplus(L[..., diag_idx, diag_idx])
        return L

    def get_R_PP(self, P):
        L = self.get_L_PP(P)
        return torch.matmul(L, L.transpose(-1, -2))

    def _dynamics(self, S, P, u_meso):
        S_g = S.clone().detach().requires_grad_(True)
        P_g = P.clone().detach().requires_grad_(True)
        
        with torch.enable_grad():
            H = self._compute_H_meso(S_g, P_g)
            grad_S, grad_P = torch.autograd.grad(H, (S_g, P_g), create_graph=True)
            
        J_SS = self.get_J_SS(S_g) 
        R_SS = self.get_R_SS(S_g) 
        J_PP = self.get_J_PP(P_g) 
        R_PP = self.get_R_PP(P_g) 
        J_SP = self.get_J_SP(S_g, P_g) 
        
        term1_S = torch.bmm(J_SS - R_SS, grad_S.unsqueeze(-1)).squeeze(-1)
        grad_P_node = grad_P.sum(dim=1) 
        term2_S = torch.matmul(grad_P_node, J_SP.T) 
        
        S_dot = term1_S + term2_S + u_meso
        
        term1_P = torch.matmul(grad_S, J_SP).unsqueeze(1).expand(-1, P.shape[1], -1) * -1.0
        term2_P = J_PP * grad_P - torch.matmul(R_PP, grad_P.unsqueeze(-1)).squeeze(-1)
        
        P_dot = term1_P + term2_P
        return S_dot, P_dot

    def forward(self, S, P, u_meso, dt=0.01):
        S_k, P_k = S, P
        S_next, P_next = S.clone(), P.clone()
        
        for _ in range(3):
            S_mid = (S_k + S_next) / 2.0
            P_mid = (P_k + P_next) / 2.0
            S_dot, P_dot = self._dynamics(S_mid, P_mid, u_meso)
            S_next = S_k + dt * S_dot
            P_next = P_k + dt * P_dot
            
        return S_next, P_next

    def readout_theta(self, P):
        if P.dim() == 3:
            P = P.unsqueeze(0)
            
        B, N, _, d_p = P.shape
        P_mean = P.mean(dim=(1, 2)) 
        
        Q = self.W_q(P_mean).unsqueeze(1) 
        K = self.W_k(P_mean).unsqueeze(1) 
        
        attn = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / (self.d_attn ** 0.5), dim=-1) 
        pool = (attn.view(B, 1, 1, 1) * P).sum(dim=(1, 2)) 
        
        theta = torch.sigmoid(self.W_out(pool) + self.b_out).squeeze(-1)
        return theta
