import numpy as np
from skopt import Optimizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


# ===============================
# LOAD DATA
# ===============================
def load_data():

    X = np.load("data/initial_inputs.npy")
    Y = np.load("data/initial_outputs.npy")

    x_new = np.array([
        [0.332155, 0.581140, 0.823007, 0.112934, 0.377212],
        [0.181818, 0.636363, 0.909090, 0.727272, 0.911111],
        [0.934962, 0.122554, 0.645815, 0.135555, 0.173420],
        [0.156301, 0.112708, 0.168904, 0.968427, 0.027408],
        [0.991695, 0.792579, 0.622739, 0.148755, 0.091244],
        [0.440651, 0.106684, 0.459520, 0.336498, 0.149086],
        [0.406582, 0.080824, 0.466373, 0.254926, 0.120558],
        [0.643477, 0.085366, 0.733772, 0.944401, 0.010025],
        [0.798931, 0.864934, 0.808448, 0.852478, 0.838243],
        [0.728186, 0.154692, 0.732551, 0.693996, 0.056401]
    ])

    y_new = np.array([
        -1.4456992800852728, -1.6480457919129663, -1.4653318485236781,
        -1.243957320498449, -1.6820172873562675, -0.9912385974245732,
        -1.0353042300144295, -0.8397919285145982, -1.8886231662500326,
        -0.7402658397514389
    ])

    X = np.vstack([X, x_new])
    Y = np.append(Y, y_new)

    return X, Y


# ===============================
# BIAS + PENALTY
# ===============================
weights = np.array([0.2, 0.2, 0.2, 0.2, 2.5])


def boundary_penalty(x):
    x = np.array(x)
    return np.sum((x < 0.1) | (x > 0.9)) * 0.3


# ===============================
# INIT OPTIMIZER
# ===============================
def create_optimizer(dim):

    opt = Optimizer(
        dimensions=[(0.0, 1.0)] * dim,
        base_estimator="GP",
        acq_func="EI",
        random_state=42
    )

    return opt


# ===============================
# MAIN LOOP
# ===============================
def run_bo(X, Y, iterations=150):

    dim = X.shape[1]
    opt = create_optimizer(dim)

    # seed optimizer
    for x, y in zip(X, Y):
        opt.tell(x.tolist(), -y)

    best_value = np.max(Y)

    for it in range(iterations):

        # =========================
        # BO suggestion
        # =========================
        x_next = np.array(opt.ask())

        # =========================
        # biased exploration (x5 focus)
        # =========================
        if np.random.rand() < 0.4:
            x_next = np.random.uniform(0, 1, dim)

            x_next[4] = np.random.beta(5, 1.5)

        x_next = np.clip(x_next, 0.05, 0.95)

        # =========================
        # REALISTIC EVALUATION (proxy)
        # =========================
        idx = np.argmin(np.linalg.norm((X - x_next) * weights, axis=1))
        y_next = Y[idx]

        # bias toward feature x5
        y_next += 0.15 * x_next[4]
        y_next -= boundary_penalty(x_next)

        opt.tell(x_next.tolist(), -y_next)

        if y_next > best_value:
            best_value = y_next

        # =========================
        # clustering exploitation
        # =========================
        X_all = np.array(opt.Xi)
        Y_all = -np.array(opt.yi)

        if len(X_all) > 20:

            top = np.percentile(Y_all, 80)
            good_X = X_all[Y_all >= top]

            if len(good_X) > 5:

                x5 = good_X[:, 4].reshape(-1, 1)

                nn = NearestNeighbors(n_neighbors=3).fit(x5)
                dists, _ = nn.kneighbors(x5)

                eps = max(np.mean(dists[:, -1]), 1e-3)

                labels = DBSCAN(eps=eps, min_samples=3).fit_predict(x5)

                clusters = [c for c in set(labels) if c != -1]

                if clusters:
                    c = np.random.choice(clusters)
                    pts = good_X[labels == c]

                    center = pts.mean(axis=0)
                    x_boost = center + np.random.normal(0, 0.02, dim)
                    x_boost = np.clip(x_boost, 0.05, 0.95)

                    idx = np.argmin(np.linalg.norm((X - x_boost) * weights, axis=1))
                    y_boost = Y[idx]

                    y_boost += 0.15 * x_boost[4]
                    y_boost -= boundary_penalty(x_boost)

                    opt.tell(x_boost.tolist(), -y_boost)

                    if y_boost > best_value:
                        best_value = y_boost

        if it % 10 == 0 or it == 1:
            print(f"Iter {it} | Best: {best_value:.6f}")

    return opt


# ===============================
# RUN
# ===============================
def main():

    X, Y = load_data()

    opt = run_bo(X, Y)

    best_idx = np.argmin(opt.yi)

    print("\n=== FINAL RESULT ===")
    print("Best value:", -opt.yi[best_idx])
    print("Best point:", opt.Xi[best_idx])


if __name__ == "__main__":
    main()
