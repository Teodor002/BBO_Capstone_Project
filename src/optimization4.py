import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler


# =========================
# LOAD DATA
# =========================
def load_data():
    """
    Load initial dataset and append additional samples.
    """

    X_pool = np.load("data/initial_inputs.npy").astype(np.float32)
    Y_pool = np.load("data/initial_outputs.npy").astype(np.float32)

    x_new = np.array([
        [0.381293, 0.783553, 0.172293, 0.692234],
        [0.357143, 0.142857, 0.642867, 0.071426],
        [0.452971, 0.368899, 0.342044, 0.400636],
        [0.597108, 0.450640, 0.098552, 0.461003],
        [0.601770, 0.446761, 0.105367, 0.464627],
        [0.316580, 0.365178, 0.373058, 0.787651],
        [0.590384, 0.476592, 0.256922, 0.458210],
        [0.589565, 0.443414, 0.103389, 0.454275],
        [0.539635, 0.390203, 0.082370, 0.474757],
        [0.472971, 0.348899, 0.372044, 0.420636],
        [0.452971, 0.378899, 0.332044, 0.410636]
    ])

    y_new = np.array([
        -14.5022899,
        -11.8631968,
        -0.1153165,
        -8.3540116,
        -8.3010083,
        -9.6075671,
        -5.1909497,
        -7.9971127,
        -8.0116618,
        -0.5717622,
        -0.32370204704882566
    ])

    X = np.vstack([X_pool, x_new])
    y = np.append(Y_pool, y_new)

    return X, y


# =========================
# MODEL
# =========================
def create_gp():
    """
    Gaussian Process with stable kernel.
    """

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * \
             Matern(length_scale=0.5, nu=2.5) + \
             WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-1))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=False
    )

    return gp


# =========================
# ACQUISITION FUNCTION
# =========================
def acquisition_ucb(mu, sigma, beta=4.0):
    """
    Upper Confidence Bound acquisition.
    """
    sigma = np.maximum(sigma, 1e-9)
    return mu + beta * sigma


# =========================
# BO LOOP
# =========================
def bayesian_optimization(X, y, n_iter=200):

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    gp = create_gp()
    dim = X.shape[1]

    for i in range(n_iter):

        # scale data
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # fit GP
        gp.fit(X_scaled, y_scaled)

        # generate candidates
        X_candidates = np.random.uniform(0, 1, size=(5000, dim))
        X_cand_scaled = scaler_X.transform(X_candidates)

        # predict
        mu, sigma = gp.predict(X_cand_scaled, return_std=True)

        # acquisition
        acq = acquisition_ucb(mu, sigma, beta=4.0)

        # select next point
        idx = np.argmax(acq)
        x_next = X_candidates[idx]

       
        y_next_scaled = mu[idx]
        y_next = scaler_y.inverse_transform([[y_next_scaled]])[0, 0]

        # update dataset
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

        print(f"Iter {i+1}/{n_iter} | Best: {np.max(y):.6f}")

    return X, y


# =========================
# MAIN
# =========================
def main():

    np.random.seed(42)

    X, y = load_data()

    X, y = bayesian_optimization(X, y)

    best_idx = np.argmax(y)

    print("\n=== FINAL ===")
    print("Best value:", y[best_idx])
    print("Best x:", X[best_idx])


if __name__ == "__main__":
    main()
