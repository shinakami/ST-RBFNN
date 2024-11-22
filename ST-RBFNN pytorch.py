import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# Radial Basis Function (RBF) kernel
def rbf_kernel(x, centers, gamma):
    """
    Compute the RBF kernel output between x and centers.
    """
    # Compute squared Euclidean distance
    distance = torch.cdist(x, centers)  # GPU-accelerated pairwise distance
    return torch.exp(-gamma * (distance ** 2))

# Spatio-Temporal Radial Basis Function Neural Network (ST-RBFNN)
class ST_RBFNN(nn.Module):
    def __init__(self, num_centers, gamma=1.0, activation='relu', device='cuda'):
        super(ST_RBFNN, self).__init__()
        self.num_centers = num_centers
        self.gamma = gamma
        self.device = device
        self.centers = None
        self.weights = None

        # Define activation function
        self.activation_func = self._get_activation(activation)

    def _get_activation(self, activation):
        """
        Return the activation function based on the specified name.
        """
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def fit(self, X, y, epochs=100, lr=0.01):
        """
        Train the RBF neural network using gradient descent.
        """
        #K-means to check centers
        kmeans = KMeans(n_clusters=self.num_centers, random_state=42)
        kmeans.fit(X.cpu().numpy())  # turn data to numpy to fit k-means
        self.centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.device)
        
        # Randomly initialize centers
        #random_idx = torch.randperm(X.size(0))[:self.num_centers]
        #self.centers = X[random_idx].to(self.device)  # Transfer centers to GPU


        # Initialize weights randomly
        self.weights = nn.Parameter(torch.randn(self.num_centers, 1, device=self.device))

        # Optimizer
        optimizer = optim.Adam([self.weights], lr=lr)

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Compute RBF output
            RBF_output = rbf_kernel(X, self.centers, self.gamma)

            # Apply activation function
            RBF_output = self.activation_func(RBF_output)

            # Compute predictions
            predictions = RBF_output @ self.weights

            # Loss (Mean Squared Error)
            loss = torch.mean((predictions - y) ** 2)

            # Backpropagation
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def predict(self, X):
        """
        Predict using the trained RBF neural network.
        """
        # Compute RBF output
        RBF_output = rbf_kernel(X, self.centers, self.gamma)

        # Apply activation function
        RBF_output = self.activation_func(RBF_output)

        # Compute predictions
        return RBF_output @ self.weights

# Generate synthetic data
def generate_data(n_samples=2000):
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, n_samples)   # Time feature

    # Generate features with added noise
    x = np.sin(t) + np.random.normal(scale=0.1, size=n_samples)  # Spatial feature (x) with noise
    y = np.cos(t) + np.random.normal(scale=0.1, size=n_samples)  # Spatial feature (y) with noise

    
    # Target variable
    pressure = np.sin(t) + 0.5 * np.cos(2 * t) + np.random.normal(scale=0.1, size=n_samples)  # Pressure as target
    
    # Combine x, y, t, into one array for input features
    X = np.vstack((x, y, t)).T  # Shape: (n_samples, 3)
    
    return X, pressure

# Main workflow
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Generate data
X, y = generate_data(n_samples=2000)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)  # Add extra dimension for target

# Train-test split
split_ratio = 0.7
train_size = int(split_ratio * len(X))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Initialize model
model = ST_RBFNN(num_centers=100, gamma=1.75, device=device, activation='relu')

# Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=70000000, lr=0.01)

# Predict and evaluate
print("Evaluating the model...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Convert predictions and true values to NumPy for easier comparison
y_test_np = y_test.cpu().detach().numpy()
y_pred_test_np = y_pred_test.cpu().detach().numpy()

# Calculate overall Mean Squared Error (MSE)
mse = mean_squared_error(y_test_np, y_pred_test_np)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot test set predictions
plt.plot(np.arange(len(y)), y, label="True Pressure")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test_np)), y_pred_test_np.flatten(), label="Predicted Pressure", linestyle="--")

# Add vertical line to show train-test split
plt.axvline(x=len(y_train), color='gray', linestyle=":")

# Add MSE to the plot
plt.text(len(y_train) + 50, np.max(y), f'MSE: {mse:.4f}', fontsize=12, color='black')

# Title and labels
plt.title("ST-RBFNN Pressure Prediction (Pytorch cuda)")
plt.xlabel("Sample Index")
plt.ylabel("Pressure")
plt.legend()

# Show plot
plt.show()
