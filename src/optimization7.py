import numpy as np
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.random.seed(42)


# ===============================
# SETTINGS
# ===============================
NUM_ITER = 120
NUM_CANDIDATES = 5000
DIM = 6


# ===============================
# BLACK BOX FUNCTION
# ===============================
class BlackBox:
    """
    Multi-modal synthetic objective function.
    """

    def __init__(self, dim, seed=42):
        np.random.seed(seed)
        self.dim = dim

        self.modes = np.random.randint(2, 5)
        self.std = np.random.uniform(0.01, 0.05, size=self.modes)
        self.means = np.random.uniform(size=(self.modes, dim))
        self.amps = np.random.uniform(5, 10, size=self.modes)

    def __call__(self, x):
        x = np.atleast_2d(x)

        exp_term = -np.sum(
            (x[:, None, :] - self.means) ** 2,
            axis=2
        ) / self.std

        return np.sum(self.amps * np.exp(exp_term), axis=1)[0]


black_box = BlackBox(DIM)


# ===============================
# LOAD DATA
# ===============================
def load_data():

    X = np.load("data/initial_inputs.npy")
    Y = np.load("data/initial_outputs.npy")

    x_new = np.array([
        [0.772114, 0.633921, 0.589332, 0.982265, 0.593332, 0.332811],
        [0.057896, 0.491672, 0.257422, 0.218118, 0.420428, 0.737097],
        [0.067896, 0.471672, 0.256722, 0.216118, 0.390428, 0.677970],
        [0.452286, 0.110836, 0.480613, 0.299925, 0.176729, 0.596741],
        [0.441577, 0.095085, 0.495764, 0.326873, 0.153888, 0.643587],
        [0.817112, 0.548168, 0.103348, 0.124370, 0.728235, 0.449674],
        [0.469725, 0.106671, 0.501487, 0.318397, 0.163921, 0.627453],
        [0.471725, 0.124064, 0.407782, 0.323771, 0.202449, 0.675492],
        [0.475382, 0.085108, 0.406577, 0.347457, 0.147479, 0.626912]
    ])

    y_new = np.array([
        0.01137,
        1.3678,
        1.6581,
        1.9582,
        1.8074,
        0.01479,
        1.8142611717461834,
        2.15855551824108,
        1.6181867441831441
    ])

    X = np.vstack([X, x_new])
    Y = np.append(Y, y_new)

    return X, Y


# ===============================
# MODEL
# ===============================
def create_gp():

    kernel = C(1.0) * Matern(length_scale=np.ones(DIM), nu=2.5) + WhiteKernel(1e-6)

    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=42
    )


# ===============================
# ACQUISITION FUNCTION
# ===============================
def ucb(mu, sigma, kappa):
    sigma = np.clip(sigma, 1e-6, None)
    return mu + kappa * sigma


# ===============================
# OPTIMIZATION LOOP
# ===============================
def optimize(X, Y, black_box):

    gp = create_gp()

    best_idx = np.argmax(Y)
    best_x = X[best_idx]
    best_y = Y[best_idx]

    for it in range(NUM_ITER):

        gp.fit(X, Y)

        # exploration + exploitation
        global_c = np.random.rand(NUM_CANDIDATES // 2, DIM)

        noise = 0.05
        local_c = best_x + np.random.normal(0, noise, (NUM_CANDIDATES // 2, DIM))
        local_c = np.clip(local_c, 0, 1)

        candidates = np.vstack([global_c, local_c])

        mu, sigma = gp.predict(candidates, return_std=True)

        kappa = 2.0 * (1 - it / NUM_ITER)
        score = ucb(mu, sigma, kappa)

        idx = np.argmax(score)
        x_next = candidates[idx]

        # true evaluation
        y_next = black_box(x_next)

        X = np.vstack([X, x_next])
        Y = np.append(Y, y_next)

        if y_next > best_y:
            best_y = y_next
            best_x = x_next

        if it % 10 == 0 or it == 1:
            print(f"Iter {it} | Best = {best_y:.6f}")

    return best_x, best_y


# ===============================
# MAIN
# ===============================
def main():

    X, Y = load_data()

    best_x, best_y = optimize(X, Y, black_box)

    print("\n=== FINAL RESULT ===")
    print("Best y:", best_y)
    print("Best x:", best_x)
