#!/usr/bin/env python3


import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
import tkinter as tk
from tkinter import messagebox

matplotlib.use("TkAgg")
np.random.seed()

# ==========================================================
# SMR++ Configuration
# ==========================================================
N = 1800
W = np.linspace(-6, 6, N)
DW = W[1] - W[0]
TARGET = 0.75
STEPS = 1400
DT = 0.002

# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------
def normalize(rho):
    rho = np.maximum(rho, 0)
    return rho / np.sum(rho * DW)

def entropy(p):
    p = p[p > 1e-12]
    return -np.sum(p * np.log(p))

def diffusion_matrix(D):
    a = D * DT / DW**2
    ab = np.zeros((3, N))
    ab[0,1:] = -a
    ab[1,:]  = 1 + 2*a
    ab[2,:-1]= -a
    ab[1,0] = ab[1,-1] = 1 + a
    return ab

# ----------------------------------------------------------
# SMR++ Controllers
# ----------------------------------------------------------
def smr_controller(rho, tau, integ):
    active = np.sum(rho[np.abs(W) > tau] * DW)
    sigma = 1 - active
    e = TARGET - sigma
    integ += e * DT
    tau = max(0.02, tau + 0.18*e + 0.03*integ)
    return tau, sigma, integ

def soft_threshold(rho, lam=1.4):
    return normalize(np.exp(-lam*np.abs(W)) * rho)

def hard_cut(rho, tau=1.2):
    rho[np.abs(W) < tau] = 0
    return normalize(rho)

# ----------------------------------------------------------
# Simulation Engine
# ----------------------------------------------------------
class SMRSimulation:
    def __init__(self, method="SMR"):
        self.method = method
        self.rho = normalize(np.exp(-0.5*W**2))
        self.tau = 1.0
        self.integ = 0.0
        self.sigmas = []
        self.entropies = []

    def step(self):
        if self.method == "SMR":
            self.tau, sigma, self.integ = smr_controller(self.rho, self.tau, self.integ)
        elif self.method == "SOFT":
            self.rho = soft_threshold(self.rho)
            sigma = np.sum(self.rho[np.abs(W)<1]*DW)
        elif self.method == "HARD":
            self.rho = hard_cut(self.rho)
            sigma = np.sum(self.rho[np.abs(W)<1]*DW)
        else:
            sigma = np.sum(self.rho[np.abs(W)<1]*DW)

        # Adaptive diffusion
        D = 0.03 + 0.25*(1-sigma)
        ab = diffusion_matrix(D)
        self.rho = solve_banded((1,1), ab, self.rho)
        self.rho = normalize(self.rho)

        return sigma, entropy(self.rho)

    def run(self):
        for k in range(STEPS):
            sigma, ent = self.step()
            if k == STEPS//2:
                # Mid-run noise perturbation
                self.rho += 0.02*np.random.randn(N)
                self.rho = normalize(self.rho)
            self.sigmas.append(sigma)
            self.entropies.append(ent)
        return self.rho, np.array(self.sigmas), np.array(self.entropies)

# ----------------------------------------------------------
# GUI Notification
# ----------------------------------------------------------
def notify(msg):
    root = tk.Tk(); root.withdraw()
    messagebox.showinfo("SMR++ Status", msg)
    root.destroy()

# ----------------------------------------------------------
# Main Execution
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SMR++ Sparse Manifold Simulation")
    parser.add_argument("--silence", type=float, default=TARGET, help="Target sparsity")
    args = parser.parse_args()
    TARGET = args.silence

    methods = ["NONE","SMR","SOFT","HARD"]
    results = {}
    for m in methods:
        sim = SMRSimulation(method=m)
        rho, sigmas, entropies = sim.run()
        results[m] = (rho, sigmas, entropies)

    # ----------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------
    fig, axes = plt.subplots(3,1, figsize=(12,12))

    # Final Density Profiles
    for m in methods:
        axes[0].plot(W, results[m][0], label=m)
    axes[0].set_title("Final Manifold Density Profiles")
    axes[0].set_xlabel("Manifold Coordinate w")
    axes[0].set_ylabel("Density ρ(w)")
    axes[0].legend()
    axes[0].grid(True)

    # Sparsity Convergence
    for m in methods:
        axes[1].plot(results[m][1], label=m)
    axes[1].axhline(TARGET, ls="--", c="k", label="Target Sparsity")
    axes[1].set_title("Sparsity Convergence over Time")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Sparsity σ(t)")
    axes[1].legend()
    axes[1].grid(True)

    # Entropy / Stability
    for m in methods:
        axes[2].plot(results[m][2], label=m)
    axes[2].set_title("Entropy (Lyapunov-like Stability)")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Entropy H(t)")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------
    # Summary Metrics
    # ----------------------------------------------------------
    msg_lines = ["=== FINAL METRICS ==="]
    for m in methods:
        sigma = results[m][1][-1]
        ent = results[m][2][-1]
        msg_lines.append(f"{m:6} sigma={sigma:.4f}   entropy={ent:.4f}")
    msg = "\n".join(msg_lines)
    print("\n" + msg)
    notify(f"Simulation Completed\n\n{msg}")
