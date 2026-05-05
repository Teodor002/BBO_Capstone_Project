import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C


# =========================
# LOAD DATA
# =========================
def load_data():
    """
    Load dataset and append additional samples.
    """

    X = np.load("data/initial_inputs.npy")
    y = np.load("data/initial_outputs.npy")

    x_new = np.array([
        [0.132902, 0.658234, 0.336573],
        [0.641025, 0.051282, 0.717948],
        [0.578947, 0.368421, 0.578947],
        [0.263158, 0.000000, 0.068421],
        [0.613384, 0.435897, 0.205128],
        [0.685189, 0.109986, 0.709514],
        [0.981259, 0.992698, 0.987079],
        [0.492581, 0.611593, 0.340176]
    ])

    y_new = np.array([
        -0.044705294,
        -0.199335941,
        -0.043643130,
        -0.101132585,
        -0.118786575,
        -0.158033177,
        -0.420448636,
        -0.051024690
    ])

    X = np.vstack([X, x_new])
    y = np.concatenate([y, y_new])

    return X, y


# =========================
# MODEL
# =========================
def create_gp():
    """
    Create Gaussian Process model with stable kernel.
    """

    kernel = C(1.0) * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(1e-5)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=42
    )

    return gp


# =========================
# THOMPSON SAMPLING SEARCH
# =========================
def thompson_search(gp, dim, low, high,
                    iterations=800,
                    candidates_per_iter=8000):
    """
    Perform global search using Thompson Sampling.
    """

    best_x = None
    best_val = -np.inf

    for i in range(iterations):

        # Sample candidates within bounds
        candidates = low + (high - low) * np.random.rand(candidates_per_iter, dim)

        mu, sigma = gp.predict(candidates, return_std=True)

        # Thompson Sampling
        samples = mu + np.random.randn(len(mu)) * sigma

        idx = np.argmax(samples)

        if samples[idx] > best_val:
            best_val = samples[idx]
            best_x = candidates[idx]

        print(f"TS Iter {i+1}/{iterations} | Best: {best_val:.6f}")

    return best_x, best_val


# =========================
# LOCAL REFINEMENT
# =========================
def local_refinement(gp, x_start, low, high,
                     steps=5000,
                     step_size=0.03):
    """
    Perform local search with annealing around best point.
    """

    best_x = x_start.copy()
    best_val = -np.inf

    for step in range(steps):

        noise = np.random.normal(0, step_size, size=len(x_start))
        x_candidate = best_x + noise

        # enforce bounds
        x_candidate = np.clip(x_candidate, low, high)

        mu, sigma = gp.predict(x_candidate.reshape(1, -1), return_std=True)

        score = mu[0] + sigma[0]

        if score > best_val:
            best_val = score
            best_x = x_candidate
            step_size *= 0.995  # annealing

    return best_x, best_val


# =========================
# MAIN
# =========================
def main():

    np.random.seed(42)

    # Load data
    X, y = load_data()
    dim = X.shape[1]

    # Train GP
    gp = create_gp()
    gp.fit(X, y)

    # Define bounds
    low, high = 0.3, 0.65

    # Global search (Thompson Sampling)
    best_x, best_val = thompson_search(gp, dim, low, high)

    # Local refinement
    refined_x, refined_val = local_refinement(gp, best_x, low, high)

    # Final result
    print("\n=== FINAL RESULT ===")
    print("Best predicted value:", refined_val)
    print("Best x:", refined_x)

