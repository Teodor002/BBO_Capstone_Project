import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.decomposition import PCA
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

np.random.seed(42)


# =========================
# LOAD DATA
# =========================
def load_data():

    X = np.load("data/initial_inputs.npy")
    Y = np.load("data/initial_outputs.npy")

    x_new = np.array([
        [0.223723, 0.658334, 0.871344, 0.226152, 0.962282, 0.363382, 0.733212, 0.552312],
        [0.796252, 0.070365, 0.355697, 0.487566, 0.74052 , 0.70665 , 0.99, 0.381734],
        [0.106447, 0.115955, 0.072986, 0.088786, 0.453935, 0.751055, 0.538307, 0.943084],
        [0.08978 , 0.165955, 0.122986, 0.138786, 0.503935, 0.701055, 0.588307, 0.926417],
        [0.862437, 0.482734, 0.281869, 0.544102, 0.887490, 0.382655, 0.601902, 0.476462],
        [0.000000, 0.000000, 0.000000, 0.088786, 0.453935, 0.751055, 0.538307, 0.943084]
    ])

    y_new = np.array([
        7.4557, 7.4599, 9.6216, 9.6027, 8.1599, 9.5394698254629
    ])

    X = np.vstack([X, x_new])
    Y = np.append(Y, y_new)

    return X, Y


# =========================
# MODEL SETUP
# =========================
def create_gp(dim):

    kernel = C(1.0) * Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(1e-5)

    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=42
    )


# =========================
# MAIN
# =========================
def main():

    X, Y = load_data()

    # =========================
    # PCA (FIT ONCE)
    # =========================
    pca = PCA(n_components=5)
    X_red = pca.fit_transform(X)

    dim = X_red.shape[1]

    gp = create_gp(dim)

    best_idx = np.argmax(Y)
    best_y = Y[best_idx]
    best_x_red = X_red[best_idx]

    iters = 80
    cand = 5000
    explore = 0.15

    for it in range(iters):

        gp.fit(X_red, Y)

        # global + local search
        global_c = np.random.rand(cand // 2, dim)
        local_c = best_x_red + np.random.normal(0, 0.1, (cand // 2, dim))

        local_c = np.clip(local_c, 0, 1)

        candidates = np.vstack([global_c, local_c])

        mu, sigma = gp.predict(candidates, return_std=True)
        sigma = np.clip(sigma, 1e-8, None)

        score = mu + 2.0 * sigma

        if np.random.rand() < explore:
            x_next_red = candidates[np.random.randint(len(candidates))]
        else:
            x_next_red = candidates[np.argmax(score)]

        # back to original space
        x_next = pca.inverse_transform(x_next_red)
        x_next = np.clip(x_next, 0, 1)

        # =========================
        # TRUE DATA-BASED UPDATE
        # =========================
        # nearest neighbor proxy (realistic surrogate)
        idx = np.argmin(np.linalg.norm(X - x_next, axis=1))
        y_next = Y[idx]

        # update dataset
        X = np.vstack([X, x_next])
        Y = np.append(Y, y_next)

        # update latent space (stable)
        X_red = pca.transform(X)

        if y_next > best_y:
            best_y = y_next
            best_x_red = x_next_red

        explore *= 0.995

        print(f"Iter {it+1} | Best: {best_y:.6f}")


    print("\n=== FINAL ===")
    print("Best y:", best_y)
    print("Best x:", pca.inverse_transform(best_x_red))


