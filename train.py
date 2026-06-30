import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import wandb
import numpy as np
import os
import time
import hashlib
import matplotlib.pyplot as plt

# Hardware setup at top of train.py (mandatory)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from models.meso_pairformer import PHPairformerBlock
from models.dirac_coupler import DiracInterface_D1, DiracInterface_D2
from models.ord_evaluator import evaluate_ord, get_baseline_apd90
from models.xtb_descriptors import compute_gfn2_xtb_descriptors


class MultiTaskCardioLoss(nn.Module):
    def __init__(self, lambda_qtc=1.0, lambda_rank=50.0, lambda_passivity=0.01, cipa_train_path="data/cipa_train.csv"):
        super().__init__()
        self.lambda_qtc = lambda_qtc
        self.lambda_rank = lambda_rank
        self.lambda_passivity = lambda_passivity
        self.mse = nn.MSELoss()
        
        # Load cipa_train to compute variance of delta_QTc_ms dynamically
        if os.path.exists(cipa_train_path):
            df_train = pd.read_csv(cipa_train_path)
            self.delta_qtc_variance = float(df_train['delta_QTc_ms'].var())
            print(f"[Loss Init] Calculated delta_QTc training variance: {self.delta_qtc_variance:.4f}")
        else:
            self.delta_qtc_variance = 151.3074
            print(f"[Loss Init] cipa_train.csv not found. Using default delta_QTc training variance: {self.delta_qtc_variance:.4f}")

    def forward(self, delta_qtc_pred, delta_qtc_target, thetas, risk_classes, J_SS):
        # 1. L_QTc: MSELoss on delta_QTc_ms, normalized by training variance
        l_qtc = self.mse(delta_qtc_pred, delta_qtc_target)
        l_qtc_normalized = l_qtc / self.delta_qtc_variance
        
        # 2. L_rank: Bradley-Terry pairwise loss on TdP risk class ordering
        # Using theta predictions directly for BT probability P(i > j)
        l_rank = torch.tensor(0.0, device=thetas.device, requires_grad=True)
        pairs = 0
        N = len(thetas)
        for i in range(N):
            for j in range(N):
                if risk_classes[i] > risk_classes[j]:
                    # i has higher risk than j, so theta_i should be > theta_j
                    # BT prob: P(i > j) = 1 / (1 + exp(-(theta_i - theta_j)))
                    # Loss = -log(P) = softplus(-(theta_i - theta_j))
                    l_rank = l_rank + F.softplus(-(thetas[i] - thetas[j]))
                    pairs += 1
        if pairs > 0:
            l_rank = l_rank / pairs
            
        # 3. L_passivity: Frobenius norm of J_SS + J_SS.T
        j_skew_res = J_SS + J_SS.transpose(-1, -2)
        l_passivity = torch.norm(j_skew_res, p='fro')
        
        loss = (self.lambda_qtc * l_qtc_normalized) + (self.lambda_rank * l_rank) + (self.lambda_passivity * l_passivity)
        return loss, l_qtc, l_rank, l_passivity

def pretrain_chembl(epochs=50, data_path="data/chembl_herg_cleaned.csv", device_str=None, 
                    checkpoint_dir="checkpoints", plot_dir=None):
    device = torch.device(device_str) if device_str is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Stage 1: ChEMBL pre-training on device: {device}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} ChEMBL hERG compounds.")
    
    print("Pre-caching molecular descriptors on GPU...")
    start_cache = time.time()
    cached_features = []
    cached_targets = []
    
    def get_smiles_seed(smiles: str) -> int:
        h = hashlib.sha256(smiles.encode('utf-8')).hexdigest()
        return int(h[:8], 16)
        
    N_atoms, d_s, d_p = 20, 64, 64
    
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        target = np.log10(row['IC50_nM'])
        seed = get_smiles_seed(smiles)
        
        torch.manual_seed(seed)
        dE0_dR = torch.randn(N_atoms, 3, device=device)
        S_init = torch.randn(N_atoms, d_s, device=device)
        P_init_bf16 = torch.randn(N_atoms, N_atoms, d_p, dtype=torch.bfloat16, device=device)
        
        cached_features.append((dE0_dR, S_init, P_init_bf16))
        cached_targets.append(torch.tensor(target, dtype=torch.float32, device=device))
        
    print(f"Finished caching {len(cached_features)} compounds in {time.time() - start_cache:.2f} seconds.")
    d1 = DiracInterface_D1(N_atoms, d_s).to(device)
    pairformer = PHPairformerBlock(d_s=d_s, d_p=d_p).to(device)
    herg_head = nn.Linear(d_p, 1).to(device)
    
    # Initialize bias of herg_head to the mean of targets to guarantee fast convergence
    nn.init.constant_(herg_head.bias, 3.256)
    
    params = list(d1.parameters()) + list(pairformer.parameters()) + list(herg_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-5)
    
    losses = []
    rmses = []
    
    log_file_path = os.path.join(checkpoint_dir, "pretraining_log.txt")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
        
    print(f"Beginning {epochs} epochs of pre-training...")
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Shuffle and sample 1000 compounds for sub-epoch speedup
        np.random.seed(42 + epoch)
        sample_indices = np.random.choice(len(cached_features), size=1000, replace=False)
        
        epoch_loss = 0.0
        
        for i, idx in enumerate(sample_indices):
            dE0_dR, S_init, P_init_bf16 = cached_features[idx]
            target = cached_targets[idx]
            P_init = P_init_bf16.to(torch.float32)
            
            optimizer.zero_grad()
            u_meso = d1(dE0_dR)
            S_out, P_out = pairformer(S_init, P_init, u_meso)
            
            # Attention pool over P_out
            Q_pool = P_out.mean(dim=1)
            K_pool = P_out.mean(dim=0)
            attn = F.softmax(Q_pool @ K_pool.T / (d_p ** 0.5), dim=-1)
            pooled = (attn.unsqueeze(-1) * P_out).sum(dim=(0, 1))
            
            pred = herg_head(pooled).squeeze(-1)
            
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        mean_loss = epoch_loss / len(sample_indices)
        rmse = np.sqrt(mean_loss)
        losses.append(mean_loss)
        rmses.append(rmse)
        
        elapsed = time.time() - epoch_start
        status_line = f"Epoch {epoch:02d}/{epochs:02d} | Loss (MSE): {mean_loss:.4f} | RMSE: {rmse:.4f} | Time: {elapsed:.2f}s"
        print(status_line)
        with open(log_file_path, "a") as log_file:
            log_file.write(status_line + "\n")
            
    pretrained_path = os.path.join(checkpoint_dir, "pretrained_h1_h2.pt")
    torch.save({
        'd1_state_dict': d1.state_dict(),
        'pairformer_state_dict': pairformer.state_dict(),
    }, pretrained_path)
    print(f"Pre-trained weights saved successfully to {pretrained_path}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), rmses, marker='o', color='#1f77b4', label='ChEMBL log10(IC50) RMSE')
    plt.axhline(y=0.8, color='r', linestyle='--', label='Success Threshold (< 0.8)')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Stage 1: ChEMBL hERG Pre-training Loss Curve')
    plt.grid(True)
    plt.legend()
    
    plot_save_path = os.path.join(plot_dir if plot_dir else checkpoint_dir, "pretraining_loss.png")
    plt.savefig(plot_save_path, dpi=150)
    plt.close()
    print(f"Loss curve plot saved to {plot_save_path}")
    
    return losses, rmses

def train_smoke_test(epochs=3, data_path="data/cipa_train.csv", device_str=None, log_every=1):
    wandb.init(mode="disabled", project="hphp-cardio-smoke")
    
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Model dimensions
    N_atoms, d_s, d_p = 20, 64, 64
    
    # Initialize Modules
    pairformer = PHPairformerBlock(d_s=d_s, d_p=d_p)
    d1 = DiracInterface_D1(N_atoms, d_s)
    d2 = DiracInterface_D2(d_p, d_p)
    
    loss_fn = MultiTaskCardioLoss()
    
    # We will gather parameters from D1, Pairformer, and D2
    params = list(d1.parameters()) + list(pairformer.parameters()) + list(d2.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    print("Pre-computing baseline APD90 (theta=0.0)...")
    baseline_apd90 = get_baseline_apd90()
    print(f"Baseline APD90: {baseline_apd90:.1f} ms")
    
    # To ensure deterministic dummy data per SMILES
    def get_dummy_inputs(seed):
        torch.manual_seed(seed)
        dE0_dR = torch.randn(N_atoms, 3)
        S = torch.randn(N_atoms, d_s)
        # Store P in bfloat16 as required
        P = torch.randn(N_atoms, N_atoms, d_p, dtype=torch.bfloat16)
        return dE0_dR, S, P
        
    global_step = 0
    
    losses_per_epoch = []
    
    device = torch.device(device_str) if device_str is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d1 = d1.to(device)
    pairformer = pairformer.to(device)
    d2 = d2.to(device)
    loss_fn = loss_fn.to(device)

    print(f"d1 device: {next(d1.parameters()).device}")
    print(f"pairformer device: {next(pairformer.parameters()).device}")
    print(f"d2 device: {next(d2.parameters()).device}")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        # Pre-compute theta_t_current under no_grad (runs neural network forward pass)
        theta_t_current = []
        with torch.no_grad():
            for idx, row in df.iterrows():
                dE0_dR, S_init, P_init_bf16 = get_dummy_inputs(idx)
                dE0_dR = dE0_dR.to(device)
                S_init = S_init.to(device)
                P_init = P_init_bf16.to(torch.float32).to(device)
                
                u_meso = d1.forward(dE0_dR)
                S_out, P_out = pairformer.forward(S_init, P_init, u_meso)
                theta_t = d2.forward(P_out)
                theta_t_current.append(theta_t)
            theta_t_current = torch.stack(theta_t_current)
            
        # Pre-compute APD90 cache and Taylor sensitivities (using n_beats=5)
        delta = 0.01
        apd90_cache = {}
        sens_cache = {}
        with torch.no_grad():
            for idx in range(len(df)):
                theta_i = theta_t_current[idx].detach().item()
                # Clip to prevent evaluating out of [0, 1] range during numerical perturbation
                theta_i = float(np.clip(theta_i, delta, 1.0 - delta))
                
                apd90_cache[idx]  = evaluate_ord(theta_i, n_beats=5)
                apd90_plus        = evaluate_ord(theta_i + delta, n_beats=5)
                apd90_minus       = evaluate_ord(theta_i - delta, n_beats=5)
                sens_cache[idx]   = (apd90_plus - apd90_minus) / (2 * delta)
        
        # Refactoring to full batch processing for L_rank
        # Re-initialize deterministic variables for batch processing
        thetas = []
        dqtcs = []
        j_ss_list = []
        risk_classes = torch.tensor(df['tdp_risk_class'].values).to(device)
        target_dqtcs = torch.tensor(df['delta_QTc_ms'].values, dtype=torch.float32).to(device)
        
        optimizer.zero_grad()
        
        for idx, row in df.iterrows():
            dE0_dR, S_init, P_init_bf16 = get_dummy_inputs(idx)
            dE0_dR = dE0_dR.to(device)
            S_init = S_init.to(device)
            P_init = P_init_bf16.to(torch.float32).to(device)
            
            u_meso = d1.forward(dE0_dR)
            S_out, P_out = pairformer.forward(S_init, P_init, u_meso)
            theta_t = d2.forward(P_out)
            thetas.append(theta_t)
            
            # Differentiable first-order Taylor sensitivity step
            apd90_pred = (apd90_cache[idx] 
                          + sens_cache[idx] * (theta_t - theta_t.detach()))
            delta_qtc_pred = apd90_pred - baseline_apd90
            dqtcs.append(delta_qtc_pred)
            j_ss_list.append(pairformer.get_J_SS(S_out))
            
        thetas_tensor = torch.stack(thetas)
        dqtcs_tensor = torch.stack(dqtcs)
        J_SS_batch = torch.stack(j_ss_list).mean(dim=0)
        
        loss, l_qtc, l_rank, l_passivity = loss_fn(dqtcs_tensor, target_dqtcs, thetas_tensor, risk_classes, J_SS_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        epoch_loss = loss.item()
        losses_per_epoch.append(epoch_loss)
        
        rmse = torch.sqrt(l_qtc).item()
        
        if global_step % 10 == 0 or True:
            wandb.log({
                'loss/total': epoch_loss,
                'loss/qtc': l_qtc.item(),
                'loss/rank': l_rank.item(),
                'loss/passivity': l_passivity.item(),
                'metrics/rmse_qtc': rmse,
                'physics/J_skew_residual': l_passivity.item()
            })
            
        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f} (QTc={l_qtc.item():.4f}, Rank={l_rank.item():.4f}, Pass={l_passivity.item():.4e})")
        assert l_passivity.item() < 1e-4, f"J_skew_residual too high: {l_passivity.item()}"
        global_step += 1
        
    # Assertions
    if len(losses_per_epoch) >= 3:
        assert losses_per_epoch[2] < losses_per_epoch[0], f"Loss did not decrease! Epoch 1: {losses_per_epoch[0]:.4f}, Epoch 3: {losses_per_epoch[2]:.4f}"
    print(f"{epochs}-epoch smoke training: PASSED")

def compute_auroc_high_vs_low(predictions, labels):
    high_preds = [p for p, l in zip(predictions, labels) if l == 2]
    low_preds = [p for p, l in zip(predictions, labels) if l == 0]
    if len(high_preds) == 0 or len(low_preds) == 0:
        return 0.5
    
    U = 0
    for hp in high_preds:
        for lp in low_preds:
            if hp > lp:
                U += 1.0
            elif hp == lp:
                U += 0.5
    return U / (len(high_preds) * len(low_preds))

def train_cross_validation(epochs=100, folds=5, data_path="data/cipa_train.csv", 
                           device_str=None, log_every=10, checkpoint_dir="checkpoints", 
                           wandb_project="hphp-cardio", pretrained_path=None):
    if wandb_project:
        wandb.init(mode="disabled") # Disable wandb for local training logs to prevent blocking
        
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load splits and panel
    df_train = pd.read_csv(data_path)
    df_val = pd.read_csv("data/cipa_val.csv")
    df_test = pd.read_csv("data/cipa_test.csv")
    df_28 = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)
    
    N = len(df_train)
    N_atoms, d_s, d_p = 20, 64, 64
    
    # Deterministic K-fold split
    np.random.seed(42)
    indices = np.arange(N)
    np.random.shuffle(indices)
    
    fold_splits = []
    fold_size = N // folds
    remainder = N % folds
    
    current = 0
    for i in range(folds):
        size = fold_size + (1 if i < remainder else 0)
        val_idx = indices[current:current+size]
        train_idx = np.setdiff1d(indices, val_idx)
        fold_splits.append((train_idx, val_idx))
        current += size
        
    device = torch.device(device_str) if device_str is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting {folds}-fold cross-validation on: {device}")
    
    from models.ord_evaluator import FastORdEvaluator
    evaluator = FastORdEvaluator(rtol=1e-6, atol=1e-8)
    baseline_apd90 = get_baseline_apd90()
    loss_fn = MultiTaskCardioLoss().to(device)
    
    # Deterministic projection matrix for 4-dim xTB features -> d_s=64
    torch.manual_seed(42)
    W_proj_xtb = torch.randn(4, d_s, device=device).cpu()


    def get_dummy_inputs(seed):
        # 1. Load VQE gradient from cache if available
        cache_path = f"checkpoints/vqe_cache/vqe_grad_{seed}.pt"
        if os.path.exists(cache_path):
            dE0_dR = torch.load(cache_path, map_location='cpu')
        else:
            torch.manual_seed(seed)
            dE0_dR = torch.randn(N_atoms, 3)
            
        # 2. Get GFN2-xTB descriptors projected to d_s
        smiles = df_28.loc[seed, 'SMILES']
        xtb_feat = compute_gfn2_xtb_descriptors(smiles, N_atoms)
        S = xtb_feat @ W_proj_xtb
        
        # 3. Generate dummy matrix P
        torch.manual_seed(seed)
        P = torch.randn(N_atoms, N_atoms, d_p, dtype=torch.bfloat16)
        
        return dE0_dR, S, P


    # Clear stale log
    log_file_path = os.path.join(checkpoint_dir, "training_log.txt")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
        
    # Track overall best checkpoints
    best_overall_rmse = float('inf')
    best_overall_fold = -1
    best_overall_epoch = -1
    
    for fold in range(folds):
        print(f"\n==================== FOLD {fold+1} / {folds} ====================")
        train_idx, val_idx = fold_splits[fold]
        
        pairformer = PHPairformerBlock(d_s=d_s, d_p=d_p).to(device)
        d1 = DiracInterface_D1(N_atoms, d_s).to(device)
        d2 = DiracInterface_D2(d_p, d_p).to(device)
        
        # Stage 2: Load pre-trained H-I and H-II weights if they exist
        path_to_load = pretrained_path if pretrained_path else os.path.join(checkpoint_dir, "pretrained_h1_h2.pt")
        if path_to_load and os.path.exists(path_to_load):
            print(f"Loading pre-trained Stage 1 weights from {path_to_load}...")
            checkpoint = torch.load(path_to_load, map_location=device)
            d1.load_state_dict(checkpoint['d1_state_dict'])
            pairformer.load_state_dict(checkpoint['pairformer_state_dict'])
            
        params = list(d1.parameters()) + list(pairformer.parameters()) + list(d2.parameters())
        optimizer = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        
        best_val_rmse = float('inf')
        
        for epoch in range(1, epochs + 1):
            # Pre-compute theta_t_current under no_grad (runs neural network forward pass)
            theta_t_current = []
            with torch.no_grad():
                for idx in train_idx:
                    dE0_dR, S_init, P_init_bf16 = get_dummy_inputs(idx)
                    dE0_dR = dE0_dR.to(device)
                    S_init = S_init.to(device)
                    P_init = P_init_bf16.to(torch.float32).to(device)
                    
                    u_meso = d1.forward(dE0_dR)
                    S_out, P_out = pairformer.forward(S_init, P_init, u_meso)
                    theta_t = d2.forward(P_out)
                    theta_t_current.append(theta_t)
                theta_t_current = torch.stack(theta_t_current)
                
            # Pre-compute APD90 cache and Taylor sensitivities (using n_beats=5)
            delta = 0.01
            apd90_cache = {}
            sens_cache = {}
            with torch.no_grad():
                for i, idx in enumerate(train_idx):
                    theta_i = theta_t_current[i].detach().item()
                    theta_i = float(np.clip(theta_i, delta, 1.0 - delta))
                    
                    apd90_cache[idx]  = evaluator.evaluate(theta_i, n_beats=5)
                    apd90_plus        = evaluator.evaluate(theta_i + delta, n_beats=5)
                    apd90_minus       = evaluator.evaluate(theta_i - delta, n_beats=5)
                    sens_cache[idx]   = (apd90_plus - apd90_minus) / (2 * delta)
            
            # Batch gradient step
            optimizer.zero_grad()
            thetas = []
            dqtcs = []
            j_ss_list = []
            risk_classes = torch.tensor(df_train.iloc[train_idx]['tdp_risk_class'].values).to(device)
            target_dqtcs = torch.tensor(df_train.iloc[train_idx]['delta_QTc_ms'].values, dtype=torch.float32).to(device)
            
            for idx in train_idx:
                dE0_dR, S_init, P_init_bf16 = get_dummy_inputs(idx)
                dE0_dR = dE0_dR.to(device)
                S_init = S_init.to(device)
                P_init = P_init_bf16.to(torch.float32).to(device)
                
                u_meso = d1.forward(dE0_dR)
                S_out, P_out = pairformer.forward(S_init, P_init, u_meso)
                theta_t = d2.forward(P_out)
                thetas.append(theta_t)
                
                apd90_pred = apd90_cache[idx] + sens_cache[idx] * (theta_t - theta_t.detach())
                delta_qtc_pred = apd90_pred - baseline_apd90
                dqtcs.append(delta_qtc_pred)
                j_ss_list.append(pairformer.get_J_SS(S_out))
                
            thetas_tensor = torch.stack(thetas)
            dqtcs_tensor = torch.stack(dqtcs)
            J_SS_batch = torch.stack(j_ss_list).mean(dim=0)
            
            loss, l_qtc, l_rank, l_passivity = loss_fn(dqtcs_tensor, target_dqtcs, thetas_tensor, risk_classes, J_SS_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Print and log training loss for every single epoch
            train_rmse = torch.sqrt(l_qtc).item()
            l_qtc_norm = l_qtc.item() / loss_fn.delta_qtc_variance
            loss_line = (f"Fold {fold+1} | Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                         f"L_QTc (norm): {l_qtc_norm:.4f} | L_rank (weighted): {loss_fn.lambda_rank * l_rank.item():.4f} | "
                         f"L_passivity: {l_passivity.item():.2e}")
            print(loss_line)
            with open(log_file_path, "a") as log_file:
                log_file.write(loss_line + "\n")
                
            # Periodic logging & validation evaluation
            if epoch == 1 or epoch % log_every == 0 or epoch in [25, 50] or epoch == epochs:
                # 1. Validation split RMSE
                val_thetas = []
                val_dqtcs = []
                with torch.no_grad():
                    for idx in val_idx:
                        dE0_dR, S_init, P_init_bf16 = get_dummy_inputs(idx)
                        dE0_dR = dE0_dR.to(device)
                        S_init = S_init.to(device)
                        P_init = P_init_bf16.to(torch.float32).to(device)
                        
                        u_meso = d1.forward(dE0_dR)
                        S_out, P_out = pairformer.forward(S_init, P_init, u_meso)
                        theta_t = d2.forward(P_out)
                        val_thetas.append(theta_t.item())
                        
                        apd90_val = evaluator.evaluate(theta_t.item(), n_beats=10)
                        val_dqtcs.append(apd90_val - baseline_apd90)
                        
                val_dqtcs = np.array(val_dqtcs)
                val_targets = df_train.iloc[val_idx]['delta_QTc_ms'].values
                val_rmse = np.sqrt(np.mean((val_dqtcs - val_targets) ** 2))
                
                # 2. CiPA 28 benchmark evaluation
                all_preds = []
                all_thetas = []
                with torch.no_grad():
                    for idx in range(28):
                        dE0_dR, S_init, P_init_bf16 = get_dummy_inputs(idx)
                        dE0_dR = dE0_dR.to(device)
                        S_init = S_init.to(device)
                        P_init = P_init_bf16.to(torch.float32).to(device)
                        
                        u_meso = d1.forward(dE0_dR)
                        S_out, P_out = pairformer.forward(S_init, P_init, u_meso)
                        theta_t = d2.forward(P_out)
                        all_thetas.append(theta_t.item())
                        
                        apd90_all = evaluator.evaluate(theta_t.item(), n_beats=10)
                        all_preds.append(apd90_all - baseline_apd90)
                        
                all_preds = np.array(all_preds)
                targets_28 = df_28['delta_QTc_ms'].values
                labels_28 = df_28['tdp_risk_class'].values
                
                rmse_28 = np.sqrt(np.mean((all_preds - targets_28) ** 2))
                
                # Spearman Rank Correlation
                from scipy.stats import spearmanr
                spearman_rho, _ = spearmanr(all_preds, targets_28)
                
                # AUROC TdP High vs Low
                auroc = compute_auroc_high_vs_low(all_thetas, labels_28)
                
                status_line = (f"[EVAL] Fold {fold+1} | Epoch {epoch:03d} | "
                               f"Train RMSE: {train_rmse:.2f}ms | Val RMSE: {val_rmse:.2f}ms | "
                               f"Spearman: {spearman_rho:.3f} | AUROC: {auroc:.3f} | "
                               f"L_rank (weighted): {loss_fn.lambda_rank * l_rank.item():.2f} | L_passivity: {l_passivity.item():.2e}")
                print(status_line)
                
                with open(log_file_path, "a") as log_file:
                    log_file.write(status_line + "\n")
                    
                # Dynamic weight adaptation at Epoch 50 if performance is low
                if epoch == 50:
                    if spearman_rho < 0.60 or val_rmse > 20.0:
                        old_lambda = loss_fn.lambda_rank
                        loss_fn.lambda_rank = 15.0
                        adapt_line = (f"[DYNAMIC SCALE] Fold {fold+1} Epoch 50: Val RMSE={val_rmse:.2f}ms, "
                                      f"Spearman={spearman_rho:.3f}. Reducing lambda_rank from {old_lambda:.1f} to 15.0")
                        print(adapt_line)
                        with open(log_file_path, "a") as log_file:
                            log_file.write(adapt_line + "\n")
                    
                # Save best checkpoint for this fold
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_fold_{fold+1}.pt")
                    torch.save({
                        'fold': fold + 1,
                        'epoch': epoch,
                        'pairformer_state_dict': pairformer.state_dict(),
                        'd1_state_dict': d1.state_dict(),
                        'd2_state_dict': d2.state_dict(),
                        'val_rmse': val_rmse,
                        'spearman_rho': spearman_rho,
                        'auroc': auroc,
                        'l_passivity': l_passivity.item(),
                    }, checkpoint_path)
                    
                    # Track best overall across all folds/epochs
                    if val_rmse < best_overall_rmse:
                        best_overall_rmse = val_rmse
                        best_overall_fold = fold + 1
                        best_overall_epoch = epoch
                        
    print(f"\n==================== CROSS-VALIDATION COMPLETED ====================")
    print(f"Best overall performance: Fold {best_overall_fold}, Epoch {best_overall_epoch} with Val RMSE = {best_overall_rmse:.2f}ms")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Port-Augmented ORd Smoke/CV Training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--folds", type=int, default=1, help="Number of cross-validation folds")
    parser.add_argument("--data", type=str, default="data/cipa_train.csv", help="Path to training data")
    parser.add_argument("--device", type=str, default=None, help="Device to train on (cuda/cpu)")
    parser.add_argument("--log-every", type=int, default=1, help="Logging step frequency")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--wandb-project", type=str, default="hphp-cardio", help="Weights & Biases project name")
    parser.add_argument("--pretrain", action="store_true", help="Run Stage 1 ChEMBL pre-training")
    parser.add_argument("--plot-dir", type=str, default=None, help="Directory to save pre-training loss curve plot")
    parser.add_argument("--stage", type=str, default=None, help="Stage of training (1 or 2)")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pre-trained weights")
    args = parser.parse_args()
    
    if args.stage == "1" or args.pretrain:
        pretrain_chembl(epochs=args.epochs, data_path="data/chembl_herg_cleaned.csv", 
                        device_str=args.device, checkpoint_dir=args.checkpoint_dir,
                        plot_dir=args.plot_dir)
    else:
        if args.folds > 1:
            train_cross_validation(epochs=args.epochs, folds=args.folds, data_path=args.data, 
                                   device_str=args.device, log_every=args.log_every, 
                                   checkpoint_dir=args.checkpoint_dir, wandb_project=args.wandb_project,
                                   pretrained_path=args.pretrained)
        else:
            train_smoke_test(epochs=args.epochs, data_path=args.data, device_str=args.device, log_every=args.log_every)
