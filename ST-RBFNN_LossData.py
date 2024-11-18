import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error

# Radial Basis Function (RBF) Kernel
def rbf_kernel(x, centers, gamma):
    distance = cdist(x, centers, 'euclidean')
    return np.exp(-gamma * (distance ** 2))

# Spatio-Temporal Radial Basis Function Neural Network for missing data imputation
class ST_RBFNN:
    def __init__(self, num_centers, gamma=1.0):
        self.num_centers = num_centers
        self.gamma = gamma
        self.centers = None
        self.weights = None

    def fit(self, X, y):
        random_idx = np.random.choice(len(X), self.num_centers, replace=False)
        self.centers = X[random_idx]
        RBF_output = rbf_kernel(X, self.centers, self.gamma)
        self.weights = np.linalg.pinv(RBF_output) @ y

    def predict(self, X):
        RBF_output = rbf_kernel(X, self.centers, self.gamma)
        return RBF_output @ self.weights

# Generate sample spatio-temporal data with missing values
def generate_data_with_missing_values(n_samples=2000, missing_ratio=0.2):
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, n_samples)
    x = np.sin(t) + np.random.normal(scale=0.1, size=n_samples)
    y = np.cos(t) + np.random.normal(scale=0.1, size=n_samples)
    pressure = np.sin(t) + 0.5 * np.cos(2 * t) + np.random.normal(scale=0.1, size=n_samples)

    # Introduce missing values in pressure
    mask = np.random.rand(n_samples) > missing_ratio
    pressure_missing = np.copy(pressure)
    pressure_missing[~mask] = np.nan  # Set missing values to NaN
    X = np.vstack((x, y, t)).T
    return X, pressure_missing, mask, pressure  # Return full data for comparison

# Initialize data
X, pressure_missing, mask, pressure_true = generate_data_with_missing_values()

# Split data into known and missing parts
X_train = X[mask]
pressure_train = pressure_missing[mask]
X_missing = X[~mask]  # Data points with missing values

# Initialize and train ST-RBFNN model
st_rbfnn = ST_RBFNN(num_centers=50, gamma=0.5)
st_rbfnn.fit(X_train, pressure_train)

# Predict missing values
pressure_pred_missing = st_rbfnn.predict(X_missing)

# Fill the missing values in the pressure data
pressure_filled = np.copy(pressure_missing)
pressure_filled[~mask] = pressure_pred_missing

# Evaluate model performance on missing data points
mse = mean_squared_error(pressure_true[~mask], pressure_pred_missing)
missing_count = np.sum(~mask)
print(f"Mean Squared Error on Missing Data: {mse:.4f}")
print(f"Number of Missing Data Points: {missing_count}")

# Plot the results for only the missing values
plt.figure(figsize=(10, 5))
plt.scatter(np.arange(len(pressure_true))[~mask], pressure_true[~mask], label="True Pressure (Missing Data Points)", color='blue', s=5)
plt.plot(np.arange(len(pressure_filled))[~mask], pressure_filled[~mask], label="Imputed Pressure", linestyle="--", color='red')
plt.legend()
plt.title(f"ST-RBFNN Imputation of Missing Pressure Data\nMSE on Missing Data: {mse:.4f}, Missing Data Count: {missing_count}")
plt.xlabel("Sample Index (Only Missing Data Points)")
plt.ylabel("Pressure")
plt.show()