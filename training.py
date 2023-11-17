import torch
import numpy as np
import pandas as pd

# Load input data from CSV file
input_data = pd.read_csv('csvs/only_data2.csv')
input_data = input_data.values.astype(np.float32)

# Load output data from CSV file
output_data = pd.read_csv('csvs/only_output.csv')
output_data = output_data.values.astype(np.float32)


# Define the neural network model
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(60, 128)
        self.fc2 = torch.nn.Linear(128, 3)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Create the neural network model
model = NeuralNetwork()

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the neural network model
for epoch in range(100):
    inputs = torch.from_numpy(input_data)
    targets = torch.from_numpy(output_data)

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch:', epoch, 'Loss:', loss.item())

# Save the trained model
torch.save(model.state_dict(), 'trained_model2.pt')

"""this is a new try 1"""
# Load the trained model
# model = torch.load('trained_model1.pt')

# Get the input data from the terminal
input_str = input("Enter the input data (separated by spaces): ")
input_data = np.array([np.double(x) for x in input_str.split()])
# Convert the input data to a tensor
inputs = torch.from_numpy(input_data)
# Get the output of the model
model = model.double()
outputs = model(inputs.double())

# Print the output data to the terminal

print(torch.detach(outputs).numpy())

# print("Output data:", outputs.numpy())
