import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.dataset import build_datasets
from src.models import SpecDAE, SpecCNN, count_trainable
from src.optimizers.kfac import KFAC
from src.optimizers.mfngd_dual import DualNGD, ExactCGNGD
from src.utils import compute_metrics, EarlyStopper
from src.profiling import SyncTimer

class CUDABenchmarkLoader:
    def __init__(self, dataset, batch_size, is_train, cfg, device):
        self.clean = dataset.clean.to(device)
        self.labels = dataset.labels.to(device)
        self.batch_size = batch_size
        self.is_train = is_train
        self.device = device
        self.cfg = cfg
        self.N = self.clean.shape[0]
        
        if not is_train:
            g = torch.Generator(device=device).manual_seed(dataset.noise_seed)
            self.fixed_noisy = self.clean + cfg.NOISE_FACTOR * torch.randn(self.clean.shape, generator=g, device=device)

    def get_batches(self, epoch, dae=None, yield_clean=False):
        if self.is_train:
            indices = torch.randperm(self.N, device=self.device)
            g = torch.Generator(device=self.device).manual_seed(self.cfg.RANDOM_SEED + epoch)
        else:
            indices = torch.arange(self.N, device=self.device)
            
        for i in range(0, self.N, self.batch_size):
            idx = indices[i:i + self.batch_size]
            if self.is_train and len(idx) < self.batch_size:
                continue
                
            labels = self.labels[idx]
            clean_b = self.clean[idx]
            
            if self.is_train:
                noisy = clean_b + self.cfg.NOISE_FACTOR * torch.randn(clean_b.shape, generator=g, device=self.device)
            else:
                noisy = self.fixed_noisy[idx]
                
            if dae is not None:
                with torch.no_grad():
                    xb = dae(noisy)
            else:
                xb = noisy
                
            if yield_clean:
                yield xb, clean_b
            else:
                yield xb, labels


def train_and_eval(optimizer_type, train_loader, val_loader, test_loader, dae, cfg, device, pos_weight):
    torch.manual_seed(cfg.RANDOM_SEED)
    
    cnn = SpecCNN(norm=cfg.NORM, gn_groups=cfg.GN_GROUPS, dropout=cfg.DROPOUT).to(device)
    cnn_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    if optimizer_type == "Adam":
        opt = torch.optim.Adam(cnn.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    elif optimizer_type == "SGD":
        opt = torch.optim.SGD(cnn.parameters(), lr=cfg.LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    elif optimizer_type == "KFAC":
        opt = KFAC(cnn, lr=cfg.LEARNING_RATE, damping=cfg.KFAC_DAMPING, invert_freq=cfg.KFAC_INVERT_FREQ)
    elif optimizer_type == "TR-MFNGD-ExactCG":
        opt = ExactCGNGD(cnn, lr=cfg.LEARNING_RATE, cg_iters=cfg.MFNGD_CG_ITERS, damping=cfg.MFNGD_INIT_DAMPING, delta=cfg.MFNGD_MAX_KL, dropout=False)
    elif optimizer_type == "TR-MFNGD-Dual":
        opt = DualNGD(cnn, lr=cfg.LEARNING_RATE, damping=cfg.MFNGD_INIT_DAMPING, delta=cfg.MFNGD_MAX_KL, solve_dtype=torch.float32, dropout=False)

    print(f"\n[Benchmarking Optimizer: {optimizer_type}] (Parameters: {count_trainable(cnn):,})")
    
    early_stopper = EarlyStopper(patience=cfg.EARLY_STOP_PATIENCE, mode="max")
    
    with SyncTimer(device) as timer:
        for epoch in range(cfg.CNN_EPOCHS):
            cnn.train()
            for xb, yb in train_loader.get_batches(epoch, dae):
                if "TR-MFNGD" in optimizer_type:
                    opt.step(xb, yb, pos_weight)
                else:
                    opt.zero_grad()
                    logits = cnn(xb)
                    loss = cnn_criterion(logits, yb)
                    loss.backward()
                    opt.step()

            cnn.eval()
            val_preds, val_probs, val_targets = [], [], []
            with torch.no_grad():
                for xb, yb in val_loader.get_batches(epoch, dae):
                    logits = cnn(xb)
                    probs = torch.sigmoid(logits)
                    val_probs.extend(probs.cpu().numpy().flatten())
                    val_preds.extend((probs >= 0.5).cpu().numpy().astype(int).flatten())
                    val_targets.extend(yb.cpu().numpy().flatten().astype(int))
                    
            val_metrics = compute_metrics(np.array(val_targets), np.array(val_preds), np.array(val_probs))
            print(f"  Epoch {epoch+1:02d} | Val AUPRC: {val_metrics['AUPRC']:.4f} | Val F1: {val_metrics['F1 Score']:.4f}")
            
            if early_stopper.step(val_metrics['AUPRC'], cnn, epoch):
                print(f"  -> Early stopping triggered at epoch {epoch+1}")
                break

    wall_time = timer.elapsed
    early_stopper.restore(cnn)
    cnn.eval()
    
    test_preds, test_probs, test_targets = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader.get_batches(0, dae):
            logits = cnn(xb)
            probs = torch.sigmoid(logits)
            test_probs.extend(probs.cpu().numpy().flatten())
            test_preds.extend((probs >= 0.5).cpu().numpy().astype(int).flatten())
            test_targets.extend(yb.cpu().numpy().flatten().astype(int))

    metrics = compute_metrics(np.array(test_targets), np.array(test_preds), np.array(test_probs))
    metrics["Wall Time (s)"] = wall_time
    metrics["Epochs"] = early_stopper.best_epoch + 1 if early_stopper.best_epoch != -1 else cfg.CNN_EPOCHS
    return metrics

def run_benchmark():
    cfg = Config()
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on device: {device}")
    
    datasets, _ = build_datasets(cfg)
    
    train_loader = CUDABenchmarkLoader(datasets["train"], cfg.BATCH_SIZE, True, cfg, device)
    val_loader = CUDABenchmarkLoader(datasets["val"], cfg.BATCH_SIZE, False, cfg, device)
    test_loader = CUDABenchmarkLoader(datasets["test"], cfg.BATCH_SIZE, False, cfg, device)

    y_tr = datasets["train"].labels.numpy()
    num_pos = np.sum(y_tr == 1)
    num_neg = np.sum(y_tr == 0)
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)

    dae = SpecDAE().to(device)
    dae_opt = torch.optim.Adam(dae.parameters(), lr=cfg.DAE_LR)
    dae_criterion = nn.MSELoss()
    dae.train()
    print("\nPre-training DAE on noisy ECG signals...")
    for epoch in range(cfg.DAE_EPOCHS):
        for noisy, clean in train_loader.get_batches(epoch, dae=None, yield_clean=True):
            dae_opt.zero_grad()
            loss = dae_criterion(dae(noisy), clean)
            loss.backward()
            dae_opt.step()
    dae.eval()

    optimizers = ["Adam", "SGD", "KFAC", "TR-MFNGD-ExactCG", "TR-MFNGD-Dual"]
    results = {}
    for opt_name in optimizers:
        results[opt_name] = train_and_eval(opt_name, train_loader, val_loader, test_loader, dae, cfg, device, pos_weight)

    base_time = results["SGD"]["Wall Time (s)"]

    print("\n================================ UNIFIED BENCHMARK COMPARISON ================================")
    print(f"{'Metric':<15} | {'Adam':<10} | {'SGD':<10} | {'KFAC':<10} | {'TR-MFNGD-ExactCG':<17} | {'TR-MFNGD-Dual':<15}")
    print("-" * 92)
    
    metrics_display = [("AUPRC", "%"), ("Accuracy", "%"), ("Precision", "%"), ("Recall", "%"), ("F1 Score", "%"), ("Wall Time (s)", "s")]

    for m_key, unit in metrics_display:
        v_adam = f"{results['Adam'][m_key]*100:.2f}%" if unit == "%" else f"{results['Adam'][m_key]:.2f}s"
        v_sgd = f"{results['SGD'][m_key]*100:.2f}%" if unit == "%" else f"{results['SGD'][m_key]:.2f}s"
        v_kfac = f"{results['KFAC'][m_key]*100:.2f}%" if unit == "%" else f"{results['KFAC'][m_key]:.2f}s"
        v_ecg = f"{results['TR-MFNGD-ExactCG'][m_key]*100:.2f}%" if unit == "%" else f"{results['TR-MFNGD-ExactCG'][m_key]:.2f}s"
        v_dual = f"{results['TR-MFNGD-Dual'][m_key]*100:.2f}%" if unit == "%" else f"{results['TR-MFNGD-Dual'][m_key]:.2f}s"
        print(f"{m_key:<15} | {v_adam:<10} | {v_sgd:<10} | {v_kfac:<10} | {v_ecg:<17} | {v_dual:<15}")

    print("-" * 92)
    print("RELATIVE CONVERGENCE INDEX (Synchronized Timers):")
    print("-" * 92)
    
    for opt_name in optimizers:
        rel_time = results[opt_name]["Wall Time (s)"] / base_time
        epochs = results[opt_name]["Epochs"]
        print(f"[{opt_name:<16}] Rel-Time Index: {rel_time:.2f}x | Converged Epochs: {epochs}")

if __name__ == "__main__":
    run_benchmark()
