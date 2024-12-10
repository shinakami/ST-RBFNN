import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Radial Basis Function (RBF) kernel
def rbf_kernel(x, centers, gamma):
    """
    Compute the RBF kernel output between x and centers.

    Parameters:
        x (torch.Tensor): Input data of shape (n_samples, n_features).
        centers (torch.Tensor): RBF kernel centers of shape (n_centers, n_features).
        gamma (float): Kernel parameter controlling the width of the Gaussian.

    Returns:
        torch.Tensor: RBF kernel output of shape (n_samples, n_centers).
    """
    # Initialize StandardScaler and fit it on x and centers together
    scaler = StandardScaler()
    combined = torch.cat((x, centers), dim=0).cpu().numpy()  # Move to CPU for scaling
    scaler.fit(combined)
    
    # Apply the same scaling transformation
    x = torch.tensor(scaler.transform(x.cpu().numpy()), device=x.device)
    centers = torch.tensor(scaler.transform(centers.cpu().numpy()), device=x.device)
    # Ensure both inputs are on the same device
    centers = centers.to(x.device)
    # Compute squared Euclidean distance
    distance = torch.cdist(x, centers)  # GPU-accelerated pairwise distance
    return torch.exp(-gamma * (distance ** 2))

# HDST-RBFNN class (with fixes)
class HDST_RBFNN(nn.Module):
    def __init__(self, num_centers, gamma=1.0, dnn_hidden_layers=[100, 50], activation='relu', dropout_prob=0.3, device='cuda'):
        super(HDST_RBFNN, self).__init__()
        self.num_centers = num_centers
        self.gamma = gamma
        self.device = device
        self.centers = None

        # Activation function
        self.activation_func = self._get_activation(activation)

        # DNN layers
        dnn_layers = []
        input_size = num_centers
        for hidden_size in dnn_hidden_layers:
            dnn_layers.append(nn.Linear(input_size, hidden_size))
            dnn_layers.append(self._get_activation(activation))
            dnn_layers.append(nn.Dropout(p=dropout_prob))
            input_size = hidden_size
        dnn_layers.append(nn.Linear(input_size, 1))
        self.dnn = nn.Sequential(*dnn_layers).to(device) 
        self.dnn.apply(self.weights_init)


    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
    def weights_init(self, m):
        """
        Initialize the weights using He initialization for the linear layers.
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def fit(self, X, y, epochs=100, lr=0.01):
        # Initialize centers using KMeans
        kmeans = KMeans(n_clusters=self.num_centers, random_state=42)
        kmeans.fit(X.cpu().numpy())
        self.centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.device)

        # Optimizer
        #optimizer = optim.Adam(self.dnn.parameters(), lr=lr, weight_decay=1e-5)
        optimizer = optim.RMSprop(self.dnn.parameters(), lr=lr, weight_decay=1e-5)


        criterion = nn.MSELoss()

        # Save losses
        losses = []

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Compute RBF output
            RBF_output = rbf_kernel(X, self.centers, self.gamma)

            # Apply activation function
            RBF_output = self.activation_func(RBF_output)

            # Pass through DNN
            predictions = self.dnn(RBF_output)

            # Loss
            loss = criterion(predictions, y)
            losses.append(loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
            
        # Visualize losses
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()

    def predict(self, X):
        RBF_output = rbf_kernel(X, self.centers, self.gamma)
        RBF_output = self.activation_func(RBF_output)
        return self.dnn(RBF_output)

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
print(f"mode: {device}")

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
model = HDST_RBFNN(num_centers=train_size, gamma=0.05, dnn_hidden_layers=[len(X), len(X), len(X)-train_size], device=device, activation='relu')

# Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=3000, lr=0.001)

# Predict and evaluate
print("Evaluating the model...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Convert predictions and true values to NumPy for easier comparison
y_test_np = y_test.cpu().detach().numpy()
y_pred_test_np = y_pred_test.cpu().detach().numpy()

# Calculate overall Mean Squared Error (MSE)
mse = mean_squared_error(y_test_np, y_pred_test_np)
rmse = np.sqrt(mse)
# Plot the results
plt.figure(figsize=(10, 6))
plt.get_current_fig_manager().full_screen_toggle()
# Plot test set predictions
plt.plot(np.arange(len(y)), y, label="True Pressure")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test_np)), y_pred_test_np.flatten(), label="Predicted Pressure", linestyle="--")

# Add vertical line to show train-test split
plt.axvline(x=len(y_train), color='gray', linestyle=":")

# Add MSE to the plot
plt.text(len(y_train) + 50, np.max(y), f'MSE: {mse:.4f}', fontsize=12, color='black')

# Title and labels
plt.title("HDST-RBFNN Pressure Prediction (Pytorch cuda)")
plt.xlabel("Sample Index")
plt.ylabel("Pressure")
plt.legend()

plt.savefig('HDST-RBFNN Pressure Prediction (Pytorch cuda)', dpi=100)
# Show plot
plt.show()


plt.close()
