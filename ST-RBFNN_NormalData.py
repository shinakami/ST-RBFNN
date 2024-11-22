import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# RBF Kernel Function
def rbf_kernel(x, centers, gamma):
    distance = cdist(x, centers, 'euclidean')
    return np.exp(-gamma * (distance ** 2))

# Spatio-Temporal Radial Basis Function Neural Network
class ST_RBFNN:
    def __init__(self, num_centers, gamma=1.0, activation='tanh'):
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

        #K-means to check centers
        kmeans = KMeans(n_clusters=self.num_centers, random_state=42)
        kmeans.fit(X)  
        self.centers = kmeans.cluster_centers_
        
        # Initialize RBF centers (e.g., random selection)
        #random_idx = np.random.choice(len(X), self.num_centers, replace=False)
        #self.centers = X[random_idx]
        
        # Compute RBF output
        RBF_output = rbf_kernel(X, self.centers, self.gamma)
        
        # Apply nonlinear activation
        RBF_output = self.activation_func(RBF_output)
        
        # Compute weights using the pseudoinverse
        self.weights = np.linalg.pinv(RBF_output) @ y
       
    
    def predict(self, X):
        # Compute RBF output for test data
        RBF_output = rbf_kernel(X, self.centers, self.gamma)
        
        # Apply nonlinear activation
        RBF_output = self.activation_func(RBF_output)
        
        # Compute final output
        return RBF_output @ self.weights



# Generate sample spatio-temporal data (for simplicity, a sinusoidal function with noise)
def generate_data(n_samples=2000):
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, n_samples)   # Time feature

    # Generate features with added noise
    x = np.sin(t) + np.random.normal(scale=0.1, size=n_samples)  # Spatial feature (x) with noise
    y = np.cos(t) + np.random.normal(scale=0.1, size=n_samples)  # Spatial feature (y) with noise
    #pm25 = 50 + 10 * np.sin(0.5 * t) + np.random.normal(scale=5, size=n_samples)  # PM2.5 with some periodicity and noise
    #temperature = 20 + 5 * np.cos(0.5 * t) + np.random.normal(scale=2, size=n_samples)  # Temperature with periodicity and noise
    
    # Target variable
    pressure = np.sin(t) + 0.5 * np.cos(2 * t) + np.random.normal(scale=0.1, size=n_samples)  # Pressure as target
    
    # Combine x, y, t, pm25, and temperature into one array for input features
    #X = np.vstack((x, y, t, pm25, temperature)).T  # Shape: (n_samples, 5)
    X = np.vstack((x, y, t)).T  # Shape: (n_samples, 3)
    
    return X, pressure
# Initialize data
X, pressure = generate_data()

# Split data into training and testing sets
split_ratio = 0.7
split_idx = int(len(X) * split_ratio)
X_train, pressure_train = X[:split_idx], pressure[:split_idx]
X_test, pressure_test = X[split_idx:], pressure[split_idx:]

# Initialize and train ST-RBFNN model
st_rbfnn = ST_RBFNN(num_centers=100, gamma=1.75)
st_rbfnn.fit(X_train, pressure_train)

# Predict on the test set
pressure_pred = st_rbfnn.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(pressure_test, pressure_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(pressure)), pressure, label="True Pressure")
plt.plot(np.arange(len(pressure_test)) + split_idx, pressure_pred, label="Predicted Pressure", linestyle="--")
plt.axvline(x=split_idx, color="gray", linestyle=":")
plt.legend()
plt.title("ST-RBFNN Pressure Prediction")
plt.xlabel("Sample Index")
plt.ylabel("Pressure")
plt.text(len(X_train) + 200, np.min(pressure_test), f"MSE: {mse:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.show()

