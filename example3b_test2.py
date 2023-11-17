import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate random input data in the interval [-2, 2]
torch.manual_seed(42)
input_data = torch.tensor(np.random.uniform(-2, 2, size=(100000, 1)), dtype=torch.float32)
desired_output = input_data ** 3


# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# Instantiate the model, loss function, and optimizer
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop for 100,000 epochs
for epoch in range(100000):
    # Forward pass
    predicted_output = model(input_data)

    # Compute the loss
    loss = criterion(predicted_output, desired_output)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Convert the tensor to numpy array for plotting
predicted_output = predicted_output.detach().numpy()

# Plot the results
plt.figure(figsize=(12, 6))


plt.scatter(input_data.numpy(), desired_output.numpy(), label='Desired Output')
plt.scatter(input_data.numpy(), predicted_output, label='Neural Network Output', color='orange')
plt.title('Desired Output vs Neural Network Output')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

plt.show()
