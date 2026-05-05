import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import qmc

np.random.seed(42)


# =========================
# DATA
# =========================
def load_data():

    X = np.load("data/initial_inputs.npy")
    y = np.load("data/initial_outputs.npy")

    x_new = np.array([
        [0.423567, 0.789456],
        [0.313131, 0.787879],
        [0.949495, 0.858586],
        [1.0, 0.656565],
        [0.208919, 0.671144],
        [0.111111, 0.676768],
        [0.060606, 0.686869],
        [0.084705, 0.517084],
        [0.562349, 0.761234],
        [0.650114, 0.681526],
        [0.731023, 0.732999]
    ])

    y_new = np.array([
        -4.3e-48,
        3.3e-87,
        -8.3e-109,
        -2.1e-97,
        9.9e-76,
        -4.1e-113,
        4.0e-141,
        -3.3e-86,
        -5.1e-16,
        -0.003606219797145482,
        7.715464788650221e-16
    ])

    for xi, yi in zip(x_new, y_new):
        if not any(np.allclose(xi, xj) for xj in X):
            X = np.vstack([X, xi])
            y = np.append(y, yi)

    return X, y


# =========================
# MODELS
# =========================
def create_gp(dim):

    kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-6)

    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=42
    )


def create_classifier():

    return RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )


# =========================
# LHS SAMPLING
# =========================
def lhs(n, d):
    return qmc.LatinHypercube(d=d).random(n)


# =========================
# BLACK-BOX (PLACEHOLDER)
# =========================
def black_box(x):

    return np.sin(np.sum(x))  # safe deterministic placeholder


# =========================
# MAIN OPTIMIZATION
# =========================
def main():

    X, y = load_data()
    dim = X.shape[1]

    gp = create_gp(dim)
    clf = create_classifier()

    best_idx = np.argmax(y)
    best_x = X[best_idx]
    best_y = y[best_idx]

    threshold = np.percentile(y, 70)

    num_iter = 40
    num_candidates = 5000

    for it in range(num_iter):

        # -------------------------
        # TRAIN MODELS
        # -------------------------
        gp.fit(X, y)

        labels = (y >= threshold).astype(int)
        clf.fit(X, labels)

        # -------------------------
        # CANDIDATES
        # -------------------------
        global_c = lhs(num_candidates // 2, dim)

        noise = 0.08 * (1 - it / num_iter)
        local_c = best_x + np.random.normal(0, noise, (num_candidates // 2, dim))
        local_c = np.clip(local_c, 0, 1)

        candidates = np.vstack([global_c, local_c])

        # -------------------------
        # FILTER
        # -------------------------
        probs = clf.predict_proba(candidates)[:, 1]

        mask = probs > 0.5

        if np.sum(mask) < 50:
            filtered = candidates
        else:
            filtered = candidates[mask]

        # -------------------------
        # GP PREDICTION
        # -------------------------
        mu, sigma = gp.predict(filtered, return_std=True)
        sigma = np.clip(sigma, 1e-6, None)

        kappa = 2.0 * (1 - it / num_iter)

        score = mu + kappa * sigma

        x_next = filtered[np.argmax(score)]

        # -------------------------
        # REAL EVALUATION (FIXED)
        # -------------------------
        y_next = black_box(x_next)

        # -------------------------
        # UPDATE
        # -------------------------
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

        if y_next > best_y:
            best_y = y_next
            best_x = x_next

        if it % 5 == 0 or it == 1:
            print(f"Iter {it} | Best: {best_y:.6f}")

    print("\n=== FINAL ===")
    print("Best x:", best_x)
    print("Best y:", best_y)


if __name__ == "__main__":
    main()
