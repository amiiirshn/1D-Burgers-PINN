import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Device setup: Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# Define the Physics-Informed Neural Network (PINN)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),  # Input layer: 2 inputs (x, t)
            nn.Tanh(),         # Activation function
            nn.Linear(64, 64), # Hidden layer
            nn.Tanh(),
            nn.Linear(64, 1)   # Output layer: 1 output (u)
        )

    def forward(self, x, t):
        # Combine x and t into a single input tensor
        inputs = torch.cat([x, t], dim=1)
        return self.model(inputs)

# Define problem parameters
L = 1.0  # Length of the spatial domain
T = 1.0  # Time duration
nu = 0.01  # Viscosity coefficient

# Initial and boundary conditions
def initial_condition(x):
    # Initial condition: u(x, 0) = sin(pi * x)
    return torch.sin(np.pi * x)

def boundary_condition(t):
    # Boundary condition at x=0: u(0, t) = 0
    return torch.tensor([0.0])

# Sample points in the domain
def sample_points(n_points):
    # Randomly sample n_points from the spatial and temporal domain
    x = torch.rand((n_points, 1), device=device) * L
    t = torch.rand((n_points, 1), device=device) * T
    return x.requires_grad_(True), t.requires_grad_(True)

# Compute derivatives using automatic differentiation
def compute_derivatives(u, x, t):
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return u_x, u_xx, u_t

# Define the loss function
def loss_function(model, n_points):
    # Sample random points
    x, t = sample_points(n_points)
    u = model(x, t)

    # Compute residuals for the Burgers' equation
    u_x, u_xx, u_t = compute_derivatives(u, x, t)
    residual = u_t + u * u_x - nu * u_xx
    loss_pde = torch.mean(residual**2)

    # Enforce the initial condition
    x_init = torch.linspace(0, L, 100, device=device).unsqueeze(1)
    t_init = torch.zeros_like(x_init, device=device)
    u_init_pred = model(x_init, t_init)
    u_init_true = initial_condition(x_init)
    loss_ic = torch.mean((u_init_pred - u_init_true)**2)

    # Total loss
    return loss_pde + loss_ic

# Train the PINN model
def train(model, n_epochs, n_points, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        loss = loss_function(model, n_points)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Initialize and train the model
model = PINN().to(device)
train(model, n_epochs=5000, n_points=1000, lr=0.01)

# Test the model: Predict u(x, t) at t = 0.5
x_test = torch.linspace(0, L, 100, device=device).unsqueeze(1)
t_test = torch.full_like(x_test, 0.5)  # Time = 0.5
u_pred = model(x_test, t_test).detach().cpu().numpy()

# Plot the predicted solution
plt.plot(x_test.cpu().numpy(), u_pred, label='Predicted')
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.title("Solution at t=0.5")
plt.show()
