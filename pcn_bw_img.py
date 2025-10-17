# pcn_bw_images.py
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------
# Nonlinearity
# ----------------------------
def f(x):        # top-down nonlinearity
    return np.tanh(x)

def fprime(x):
    y = np.tanh(x)
    return 1.0 - y * y

def normalize_columns(W, eps=1e-8):
    norms = np.linalg.norm(W, axis=0) + eps
    return W / norms

# ----------------------------
# Predictive Coding Network (2-level)
# ----------------------------
class PredictiveCodingNet:
    """
    Level 0 (input):      x  in R^{x_dim}
    Level 1 (features):   r1 in R^{r1_dim}, predicts x via U1 @ r1
    Level 2 (causes):     r2 in R^{r2_dim}, predicts r1 via f(U2 @ r2)

    Fast loop: infer r1, r2 by minimizing prediction errors.
    Slow loop: learn U1, U2 to reduce future errors.
    """
    def __init__(
        self,
        x_dim,
        r1_dim,
        r2_dim,
        eta_r=0.2,          # inference step
        eta_w=1e-3,         # weight learning step
        l2_act=1e-3,        # small L2 on activations
        inference_steps=60, # iterations per input
        seed=0
    ):
        rng = np.random.default_rng(seed)
        self.x_dim, self.r1_dim, self.r2_dim = x_dim, r1_dim, r2_dim

        self.U1 = normalize_columns(rng.normal(0, 1/np.sqrt(r1_dim), size=(x_dim, r1_dim)))
        self.U2 = normalize_columns(rng.normal(0, 1/np.sqrt(r2_dim), size=(r1_dim, r2_dim)))

        self.eta_r = eta_r
        self.eta_w = eta_w
        self.l2_act = l2_act
        self.inference_steps = inference_steps

    def infer(self, x, r1_init=None, r2_init=None):
        r1 = np.zeros(self.r1_dim) if r1_init is None else r1_init.copy()
        r2 = np.zeros(self.r2_dim) if r2_init is None else r2_init.copy()

        for _ in range(self.inference_steps):
            z2   = self.U2 @ r2           # pre-activation at level-1 (from level-2)
            r1td = f(z2)                  # top-down prediction for r1

            e0 = x - self.U1 @ r1         # pixel-level error
            e1 = r1 - r1td                # level-1 error

            # activation updates
            r1 += self.eta_r * (self.U1.T @ e0 - e1 - self.l2_act * r1)
            r2 += self.eta_r * (self.U2.T @ (fprime(z2) * e1) - self.l2_act * r2)

        # final compute
        z2   = self.U2 @ r2
        r1td = f(z2)
        e0 = x - self.U1 @ r1
        e1 = r1 - r1td
        return r1, r2, e0, e1

    def learn(self, x, r1=None, r2=None):
        r1, r2, e0, e1 = self.infer(x, r1, r2)
        # weight updates (simple local rules)
        self.U1 += self.eta_w * np.outer(e0, r1)
        self.U2 += self.eta_w * np.outer(fprime(self.U2 @ r2) * e1, r2)

        # keep columns well-conditioned
        self.U1 = normalize_columns(self.U1)
        self.U2 = normalize_columns(self.U2)

        loss = 0.5 * (np.dot(e0, e0) + np.dot(e1, e1))
        return loss, r1, r2

    def reconstruct(self, r1):
        return self.U1 @ r1

# ----------------------------
# Image loading / preprocessing
# ----------------------------
def load_bw_images(
    root_dir="./data/bw",
    img_size=8,
    binarize=True,
    threshold=0.5,
    max_images=None
):
    """
    Loads images (PNG/JPG) from root_dir (recursively), converts to grayscale,
    resizes to img_size x img_size, optionally binarizes, and maps to [-1, 1].
    Returns array X of shape [N, img_size*img_size].
    """
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    files = sorted(files)
    if max_images is not None:
        files = files[:max_images]
    if not files:
        raise FileNotFoundError(
            f"No images found in {root_dir}. Put a few BW/grayscale images there."
        )

    X = []
    for fp in files:
        img = Image.open(fp).convert("L")                 # grayscale 0..255
        img = img.resize((img_size, img_size), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0   # to [0,1]

        if binarize:
            arr = (arr > threshold).astype(np.float32)

        # map to [-1, 1] (helps with tanh)
        arr = arr * 2.0 - 1.0
        X.append(arr.flatten())

    X = np.stack(X, axis=0)
    return X

# ----------------------------
# Visualization helpers
# ----------------------------
def show_basis(U1, img_size=8, num=16, title="Level-1 basis (columns of U1)"):
    num = min(num, U1.shape[1])
    cols = int(np.ceil(np.sqrt(num)))
    rows = int(np.ceil(num / cols))

    plt.figure(figsize=(1.6*cols, 1.6*rows))
    for i in range(num):
        plt.subplot(rows, cols, i+1)
        # each column is a basis vector in pixel space
        patch = U1[:, i].reshape(img_size, img_size)
        # rescale to [0,1] for display
        pmin, pmax = patch.min(), patch.max()
        if pmax > pmin:
            patch_disp = (patch - pmin) / (pmax - pmin)
        else:
            patch_disp = np.zeros_like(patch)
        plt.imshow(patch_disp, cmap="gray", interpolation="nearest")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def show_reconstructions(X, net, img_size=8, k=6):
    k = min(k, len(X))
    idx = np.random.default_rng(0).choice(len(X), size=k, replace=False)

    plt.figure(figsize=(8, 2*k))
    for i, j in enumerate(idx):
        x = X[j]
        r1, r2, _, _ = net.infer(x)
        x_hat = net.reconstruct(r1)

        # back to [0,1] from [-1,1]
        def to01(v):
            v = (v + 1.0) / 2.0
            return np.clip(v, 0.0, 1.0)

        x_img     = to01(x).reshape(img_size, img_size)
        xhat_img  = to01(x_hat).reshape(img_size, img_size)

        plt.subplot(k, 2, 2*i+1)
        plt.imshow(x_img, cmap="gray", interpolation="nearest")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(k, 2, 2*i+2)
        plt.imshow(xhat_img, cmap="gray", interpolation="nearest")
        plt.title("Reconstruction")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# ----------------------------
# Train loop
# ----------------------------
def train_on_images(
    X,
    img_size=8,
    r1_dim=32,
    r2_dim=16,
    epochs=20,
    eta_r=0.2,
    eta_w=1e-3,
    l2_act=1e-3,
    inference_steps=60,
    batch_size=64,
    seed=1
):
    x_dim = img_size * img_size
    net = PredictiveCodingNet(
        x_dim=x_dim, r1_dim=r1_dim, r2_dim=r2_dim,
        eta_r=eta_r, eta_w=eta_w, l2_act=l2_act,
        inference_steps=inference_steps, seed=seed
    )

    rng = np.random.default_rng(seed)
    N = len(X)
    for ep in range(1, epochs+1):
        perm = rng.permutation(N)
        X_shuf = X[perm]

        losses = []
        for i in range(0, N, batch_size):
            batch = X_shuf[i:i+batch_size]
            for x in batch:
                loss, _, _ = net.learn(x)
                losses.append(loss)
        print(f"Epoch {ep:02d} | avg total loss: {np.mean(losses):.4f}")

    return net

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # --- user knobs ---
    IMG_SIZE   = 16     # try 8 or 16; higher is harder
    BINARIZE   = True   # for crisp black/white
    THRESH     = 0.5    # binarization threshold in [0,1]
    MAX_IMAGES = None   # or an int like 500

    # model sizes: scale roughly with IMG_SIZE
    R1_DIM = 64         # level-1 features (try 2-4x IMG_SIZE)
    R2_DIM = 32         # level-2 causes (try ~half of R1_DIM)

    # training knobs
    EPOCHS     = 20
    INFER_STEPS= 80
    ETA_R      = 0.25
    ETA_W      = 1e-3
    L2_ACT     = 1e-3
    BATCH_SIZE = 64

    # 1) Load images
    X = load_bw_images(
        root_dir="./data/bw",
        img_size=IMG_SIZE,
        binarize=BINARIZE,
        threshold=THRESH,
        max_images=MAX_IMAGES
    )

    # 2) Train the predictive-coding model
    net = train_on_images(
        X,
        img_size=IMG_SIZE,
        r1_dim=R1_DIM,
        r2_dim=R2_DIM,
        epochs=EPOCHS,
        eta_r=ETA_R,
        eta_w=ETA_W,
        l2_act=L2_ACT,
        inference_steps=INFER_STEPS,
        batch_size=BATCH_SIZE,
        seed=1
    )

    # 3) Visualize level-1 basis vectors (columns of U1)
    show_basis(net.U1, img_size=IMG_SIZE, num=min(36, R1_DIM),
               title="Level-1 basis (U1 columns)")

    # 4) Show a few reconstructions
    show_reconstructions(X, net, img_size=IMG_SIZE, k=6)

