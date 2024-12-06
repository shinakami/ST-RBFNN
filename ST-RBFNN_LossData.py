import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# RBF Kernel Function
def rbf_kernel(x, centers, gamma):
    distance = cdist(x, centers, 'euclidean')
    return np.exp(-gamma * (distance ** 2))
# Spatio-Temporal Radial Basis Function Neural Network
class ST_RBFNN:
    def __init__(self, num_centers, gamma=1.0, activation='relu'):
        self.num_centers = num_centers  # Number of RBF centers
        self.gamma = gamma              # Kernel width parameter
        self.centers = None             # RBF centers
        self.weights = None             # Weights of the output layer
        self.activation_func = self._get_activation(activation)  # Nonlinear activation
    def _get_activation(self, activation):
        """Select activation function."""
        if activation == 'relu':
            return lambda x: np.maximum(0, x)
        elif activation == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif activation == 'tanh':
            return lambda x: np.tanh(x)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    def fit(self, X, y):
        # Step 1: Use K-Means to find RBF centers
        kmeans = KMeans(n_clusters=self.num_centers, random_state=42)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        # Step 2: Compute the RBF output for all input data
        RBF_output = rbf_kernel(X, self.centers, self.gamma)

        # Step 3: Apply the activation function to the RBF output
        RBF_output = self.activation_func(RBF_output)

        # Step 4: Compute the weights using the pseudoinverse of RBF output
        self.weights = np.linalg.pinv(RBF_output) @ y
        
    def predict(self, X):
        # Compute RBF output for test data
        RBF_output = rbf_kernel(X, self.centers, self.gamma)
        
        # Apply nonlinear activation
        RBF_output = self.activation_func(RBF_output)
        
        # Compute final output
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
    X = np.vstack((x, y, t)).T  # Shape: (n_samples, 3)
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
