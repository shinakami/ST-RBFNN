import torch
import torch.nn as nn
import torch.optim as optim

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
    def __init__(self, num_centers, gamma=1.0, device='cuda'):
        super(ST_RBFNN, self).__init__()
        self.num_centers = num_centers
        self.gamma = gamma
        self.device = device
        self.centers = None
        self.weights = None

    def fit(self, X, y, epochs=100, lr=0.01):
        """
        Train the RBF neural network using gradient descent.
        """
        # Randomly initialize centers
        random_idx = torch.randperm(X.size(0))[:self.num_centers]
        self.centers = X[random_idx].to(self.device)  # Transfer centers to GPU

        # Initialize weights randomly
        self.weights = nn.Parameter(torch.randn(self.num_centers, 1, device=self.device))

        # Optimizer
        optimizer = optim.Adam([self.weights], lr=lr)

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Compute RBF output
            RBF_output = rbf_kernel(X, self.centers, self.gamma)

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

        # Compute predictions
        return RBF_output @ self.weights


# 模擬數據
torch.manual_seed(42)
X = torch.rand(100, 2).to('cuda')  # 100 個樣本，每個有 2 個特徵，放到 GPU
y = torch.sin(X[:, 0] * 2 * torch.pi) + torch.cos(X[:, 1] * 2 * torch.pi)
y = y.unsqueeze(1).to('cuda')  # y 需轉成列向量

# 初始化並訓練模型
model = ST_RBFNN(num_centers=10, gamma=5.0, device='cuda')
model.fit(X, y, epochs=200, lr=0.01)

# 測試數據預測
X_test = torch.rand(10, 2).to('cuda')  # 測試 10 筆數據
y_pred = model.predict(X_test)

print("Predictions:", y_pred.cpu().detach().numpy())

