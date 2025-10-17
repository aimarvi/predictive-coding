import numpy as np

# ----------------------------
# Utilities
# ----------------------------
def f(x):        # top-down nonlinearity
    return np.tanh(x)

def fprime(x):   # derivative of tanh
    y = np.tanh(x)
    return 1.0 - y * y

def normalize_columns(W, eps=1e-8):
    # Normalize each column to unit norm to keep weights well-conditioned
    norms = np.linalg.norm(W, axis=0) + eps
    return W / norms

# ----------------------------
# Predictive Coding Network
# ----------------------------
class PredictiveCodingNet:
    """
    Two-level hierarchical predictive coding:
        Level 0 (input):      x  in R^{x_dim}
        Level 1 (features):   r1 in R^{r1_dim}, predicts x via U1 @ r1
        Level 2 (causes):     r2 in R^{r2_dim}, predicts r1 via f(U2 @ r2)

    Fast time-scale: infer r1, r2 to minimize prediction errors.
    Slow time-scale: learn U1, U2 to reduce future errors.
    """

    def __init__(
        self,
        x_dim=64,           # e.g., 8x8 image patch
        r1_dim=32,          # level-1 features
        r2_dim=16,          # level-2 causes
        eta_r=0.2,          # inference step size
        eta_w=1e-3,         # learning rate for weights
        l2_act=1e-3,        # L2 regularization on r's
        inference_steps=50, # iterations per input for inference
        seed=0
    ):
        rng = np.random.default_rng(seed)
        self.x_dim, self.r1_dim, self.r2_dim = x_dim, r1_dim, r2_dim
        # Initialize weights (basis vectors)
        self.U1 = normalize_columns(rng.normal(0, 1/np.sqrt(r1_dim), size=(x_dim, r1_dim)))
        self.U2 = normalize_columns(rng.normal(0, 1/np.sqrt(r2_dim), size=(r1_dim, r2_dim)))

        self.eta_r = eta_r
        self.eta_w = eta_w
        self.l2_act = l2_act
        self.inference_steps = inference_steps

    def infer(self, x, r1_init=None, r2_init=None):
        """
        Run fast inference to estimate r1, r2 that minimize prediction errors:
            e0 = x - U1 r1
            r1_td = f(U2 r2)
            e1 = r1 - r1_td
        Updates (schematic gradient steps):
            r1 += U1^T e0 - e1 - l2*r1
            r2 += U2^T( f'(U2 r2) ⊙ e1 ) - l2*r2
        """
        r1 = np.zeros(self.r1_dim) if r1_init is None else r1_init.copy()
        r2 = np.zeros(self.r2_dim) if r2_init is None else r2_init.copy()

        for _ in range(self.inference_steps):
            # Top-down prediction of r1 from r2
            z2 = self.U2 @ r2
            r1_td = f(z2)

            # Errors
            e0 = x - self.U1 @ r1            # input-layer error
            e1 = r1 - r1_td                  # level-1 error

            # Inference updates (gradient-like)
            r1 += self.eta_r * (self.U1.T @ e0 - e1 - self.l2_act * r1)

            # Chain-rule factor for r2 via r1_td = f(U2 r2)
            r2 += self.eta_r * (self.U2.T @ (fprime(z2) * e1) - self.l2_act * r2)

        # Final recompute for outputs
        z2 = self.U2 @ r2
        r1_td = f(z2)
        e0 = x - self.U1 @ r1
        e1 = r1 - r1_td
        return r1, r2, e0, e1

    def learn(self, x, r1=None, r2=None):
        """
        After inference has (approximately) converged,
        update weights to reduce future errors:
            L = 1/2 ||e0||^2 + 1/2 ||e1||^2
        Gradients (descent):
            dU1 = - dL/dU1 = e0 r1^T
            dU2 = - dL/dU2 = (e1 ⊙ f'(U2 r2)) r2^T
        """
        r1, r2, e0, e1 = self.infer(x, r1, r2)

        # Weight updates
        self.U1 += self.eta_w * np.outer(e0, r1)
        self.U2 += self.eta_w * np.outer(fprime(self.U2 @ r2) * e1, r2)

        # (Optional) keep columns normalized to avoid exploding norms
        self.U1 = normalize_columns(self.U1)
        self.U2 = normalize_columns(self.U2)

        # Report losses
        loss_e0 = 0.5 * np.dot(e0, e0)
        loss_e1 = 0.5 * np.dot(e1, e1)
        return loss_e0, loss_e1, r1, r2

    def reconstruct(self, r1):
        return self.U1 @ r1

# ----------------------------
# Simple synthetic dataset
# ----------------------------
def make_synthetic_data(
    N=2000, x_dim=64, r1_dim=32, r2_dim=16, active_r2=3, noise_std=0.05, seed=42
):
    """
    Create a tiny 2-level generative world:
        r2_true: sparse causes (few active)
        r1_true = tanh(U2_true r2_true)
        x       = U1_true r1_true + noise
    The learner doesn't know U1_true/U2_true; it must discover
    basis vectors that can predict the inputs.
    """
    rng = np.random.default_rng(seed)

    U1_true = normalize_columns(rng.normal(0, 1/np.sqrt(r1_dim), size=(x_dim, r1_dim)))
    U2_true = normalize_columns(rng.normal(0, 1/np.sqrt(r2_dim), size=(r1_dim, r2_dim)))

    X = np.zeros((N, x_dim))
    R1_true = np.zeros((N, r1_dim))
    R2_true = np.zeros((N, r2_dim))

    for i in range(N):
        r2 = np.zeros(r2_dim)
        idx = rng.choice(r2_dim, size=active_r2, replace=False)
        r2[idx] = rng.normal(0.8, 0.2, size=active_r2)  # a few strong causes
        r1 = f(U2_true @ r2)
        x = U1_true @ r1 + rng.normal(0, noise_std, size=x_dim)

        R2_true[i] = r2
        R1_true[i] = r1
        X[i] = x

    return X, U1_true, U2_true, R1_true, R2_true

# ----------------------------
# Demo / Training loop
# ----------------------------
def main():
    rng = np.random.default_rng(123)

    # Sizes: treat input as an 8x8 patch (64 dims)
    x_dim, r1_dim, r2_dim = 64, 32, 16

    # Build toy dataset
    X, _, _, _, _ = make_synthetic_data(
        N=2000, x_dim=x_dim, r1_dim=r1_dim, r2_dim=r2_dim, active_r2=3, noise_std=0.05, seed=7
    )

    # Initialize predictive coding net
    net = PredictiveCodingNet(
        x_dim=x_dim, r1_dim=r1_dim, r2_dim=r2_dim,
        eta_r=0.2, eta_w=1e-3, l2_act=1e-3, inference_steps=50, seed=1
    )

    # Train
    epochs = 20
    batch_size = 64
    for ep in range(1, epochs + 1):
        # Shuffle
        idx = rng.permutation(len(X))
        X_shuf = X[idx]

        losses = []
        for i in range(0, len(X_shuf), batch_size):
            batch = X_shuf[i:i+batch_size]
            # simple online updates
            for x in batch:
                loss_e0, loss_e1, _, _ = net.learn(x)
                losses.append(loss_e0 + loss_e1)

        print(f"Epoch {ep:02d} | avg total loss: {np.mean(losses):.4f}")

    # Test on a random example
    x_test = X[rng.integers(0, len(X))]
    r1_hat, r2_hat, e0, e1 = net.infer(x_test)
    x_rec = net.reconstruct(r1_hat)

    print("\n--- TEST SAMPLE ---")
    print(f"Input norm:          {np.linalg.norm(x_test):.3f}")
    print(f"Reconstruction norm: {np.linalg.norm(x_rec):.3f}")
    print(f"Reconstruction error ||x - x_hat||: {np.linalg.norm(x_test - x_rec):.3f}")
    print(f"Level-1 error ||e1||:               {np.linalg.norm(e1):.3f}")
    print(f"Top few r2 activations (indices,val):")
    top_idx = np.argsort(-np.abs(r2_hat))[:5]
    print([(int(i), float(r2_hat[i])) for i in top_idx])

if __name__ == "__main__":
    main()

