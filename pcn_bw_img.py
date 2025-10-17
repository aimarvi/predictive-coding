# pcn_coco8_gray_patches.py
import os, glob
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
        eta_r=0.25,         # inference step size
        eta_w=1e-3,         # weight learning step
        l2_act=1e-3,        # small L2 on activations
        inference_steps=80, # iterations per input
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

    def infer(self, x, r1_init=None, r2_init=None, record=False):
        """
        Run inference to estimate r1, r2.
        If record=True, also store reconstructions (U1 @ r1) at each time step.
        Returns:
        	r1, r2, e0, e1, reconstructions (list or None)
        """
        r1 = np.zeros(self.r1_dim) if r1_init is None else r1_init.copy()
        r2 = np.zeros(self.r2_dim) if r2_init is None else r2_init.copy()
     
        recons = [] if record else None
     
        for t in range(self.inference_steps):
        	z2   = self.U2 @ r2
        	r1td = f(z2)
     
        	e0 = x - self.U1 @ r1
        	e1 = r1 - r1td
     
        	# activation updates
        	r1 += self.eta_r * (self.U1.T @ e0 - e1 - self.l2_act * r1)
        	r2 += self.eta_r * (self.U2.T @ (fprime(z2) * e1) - self.l2_act * r2)
     
        	if record:
        		recons.append(self.U1 @ r1)  # predicted image at this time step
     
        z2   = self.U2 @ r2
        r1td = f(z2)
        e0 = x - self.U1 @ r1
        e1 = r1 - r1td
     
        if record:
        	return r1, r2, e0, e1, recons
        else:
        	return r1, r2, e0, e1

    def learn(self, x, r1=None, r2=None):
        r1, r2, e0, e1 = self.infer(x, r1, r2)
        # weight updates (local rules)
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
# Data: load grayscale + extract patches
# ----------------------------
def list_images(root):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    for split in ("train", "val"):
        d = os.path.join(root, "images", split)
        for fn in sorted(glob.glob(os.path.join(d, "*"))):
            if os.path.splitext(fn)[1].lower() in exts:
                files.append(fn)
    if not files:
        raise FileNotFoundError(f"No images found under {root}/images/{{train,val}}")
    return files

def load_gray(path):
    # load as grayscale float32 in [0,1]
    im = Image.open(path).convert("L")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr  # HxW

def extract_patches(img, patch_size=16, stride=8, max_patches=None, rng=None):
    """
    img: HxW in [0,1]
    returns: [N, patch_size*patch_size] mapped to [-1,1]
    """
    H, W = img.shape
    ps = patch_size
    patches = []
    for y in range(0, H - ps + 1, stride):
        for x in range(0, W - ps + 1, stride):
            p = img[y:y+ps, x:x+ps]
            patches.append(p)
    if not patches:
        # if image smaller than patch, resize and take one patch
        p = np.array(Image.fromarray((img*255).astype(np.uint8)).resize((ps, ps), Image.BICUBIC)) / 255.0
        patches = [p]

    patches = np.stack(patches, axis=0)

    if max_patches is not None and len(patches) > max_patches:
        rng = np.random.default_rng(0) if rng is None else rng
        idx = rng.choice(len(patches), size=max_patches, replace=False)
        patches = patches[idx]

    # map to [-1,1] for tanh
    patches = patches * 2.0 - 1.0
    patches = patches.reshape(len(patches), ps*ps)
    return patches

def build_patch_dataset(
    coco8_root="coco8-grayscale",
    patch_size=16,
    stride=8,
    max_images=None,
    max_patches_per_image=128,
):
    files = list_images(coco8_root)
    if max_images is not None:
        files = files[:max_images]

    all_patches = []
    rng = np.random.default_rng(123)
    for fp in files:
        img = load_gray(fp)          # HxW in [0,1]
        patches = extract_patches(
            img,
            patch_size=patch_size,
            stride=stride,
            max_patches=max_patches_per_image,
            rng=rng
        )
        all_patches.append(patches)

    X = np.concatenate(all_patches, axis=0)
    return X  # [N, patch_size*patch_size]

# ----------------------------
# Visualization helpers
# ----------------------------
def show_basis(U1, img_size=16, num=25, title="Level-1 basis (columns of U1)"):
    num = min(num, U1.shape[1])
    cols = int(np.ceil(np.sqrt(num)))
    rows = int(np.ceil(num / cols))

    plt.figure(figsize=(1.6*cols, 1.6*rows))
    for i in range(num):
        plt.subplot(rows, cols, i+1)
        patch = U1[:, i].reshape(img_size, img_size)
        pmin, pmax = patch.min(), patch.max()
        patch_disp = (patch - pmin) / (pmax - pmin + 1e-8)
        plt.imshow(patch_disp, cmap="gray", interpolation="nearest")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def show_reconstructions(X, net, img_size=16, k=6):
    k = min(k, len(X))
    idx = np.random.default_rng(0).choice(len(X), size=k, replace=False)

    def to01(v):
        v = (v + 1.0) / 2.0
        return np.clip(v, 0.0, 1.0)

    plt.figure(figsize=(8, 2*k))
    for i, j in enumerate(idx):
        x = X[j]
        r1, r2, _, _ = net.infer(x)
        x_hat = net.reconstruct(r1)

        x_img     = to01(x).reshape(img_size, img_size)
        xhat_img  = to01(x_hat).reshape(img_size, img_size)

        plt.subplot(k, 2, 2*i+1)
        plt.imshow(x_img, cmap="gray", interpolation="nearest")
        plt.title("Original patch")
        plt.axis("off")

        plt.subplot(k, 2, 2*i+2)
        plt.imshow(xhat_img, cmap="gray", interpolation="nearest")
        plt.title("Reconstruction")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def show_timecourse(x, recons, img_size=16, step_stride=5):
    """
    Display the input and several reconstruction frames along the inference trajectory.
    """
    def to01(v):  # map [-1,1] to [0,1]
        v = (v + 1.0) / 2.0
        return np.clip(v, 0, 1)

    steps = list(range(0, len(recons), step_stride)) + [len(recons)-1]
    cols = len(steps) + 1

    plt.figure(figsize=(2*cols, 2))
    plt.subplot(1, cols, 1)
    plt.imshow(to01(x).reshape(img_size, img_size), cmap="gray")
    plt.title("Input")
    plt.axis("off")

    for i, t in enumerate(steps):
        plt.subplot(1, cols, i+2)
        plt.imshow(to01(recons[t]).reshape(img_size, img_size), cmap="gray")
        plt.title(f"t={t}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ----------------------------
# Train loop
# ----------------------------
def train_on_patches(
    X,
    patch_size=16,
    r1_dim=128,
    r2_dim=64,
    epochs=15,
    eta_r=0.25,
    eta_w=1e-3,
    l2_act=1e-3,
    inference_steps=80,
    batch_size=256,
    seed=1
):
    x_dim = patch_size * patch_size
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
    # ---- paths & patching ----
    COCO8_ROOT = "coco8-grayscale"  # expects images/train and images/val
    PATCH_SIZE = 16                 # try 8 or 16
    STRIDE     = 8                  # overlap controls variety
    MAX_IMAGES = None               # e.g., 32 if you want to limit
    MAX_PATCHES_PER_IMAGE = 128     # cap for speed

    # ---- model sizes (scale with patch size) ----
    R1_DIM = 128
    R2_DIM = 64

    # ---- training knobs ----
    EPOCHS       = 15
    INFER_STEPS  = 80
    ETA_R        = 0.25
    ETA_W        = 1e-3
    L2_ACT       = 1e-3
    BATCH_SIZE   = 256

    print("Building patch dataset from COCO8 grayscale ...")
    X = build_patch_dataset(
        coco8_root=COCO8_ROOT,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        max_images=MAX_IMAGES,
        max_patches_per_image=MAX_PATCHES_PER_IMAGE
    )
    print(f"Total patches: {len(X)} | patch dim: {PATCH_SIZE*PATCH_SIZE}")

    print("Training predictive-coding model ...")
    net = train_on_patches(
        X,
        patch_size=PATCH_SIZE,
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

    # Visualize learned bases & a few reconstructions
    show_basis(net.U1, img_size=PATCH_SIZE, num=min(36, R1_DIM),
               title="Level-1 basis (U1 columns) — COCO8 patches")
    show_reconstructions(X, net, img_size=PATCH_SIZE, k=6)

    x = X[0]  # pick a patch
    r1, r2, e0, e1, recons = net.infer(x, record=True)
    show_timecourse(x, recons, img_size=PATCH_SIZE, step_stride=10)
