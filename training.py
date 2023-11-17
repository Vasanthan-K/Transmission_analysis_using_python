import torch
import numpy as np
import pandas as pd

input_data = pd.read_csv('csvs/only_data2.csv')
input_data = input_data.values.astype(np.float32)

output_data = pd.read_csv('csvs/only_output.csv')
output_data = output_data.values.astype(np.float32)


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


model = NeuralNetwork()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

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

"""this is a new try 1"""

input_str = input("Enter the input data (separated by spaces): ")
input_data = np.array([np.double(x) for x in input_str.split()])

inputs = torch.from_numpy(input_data)

model = model.double()
outputs = model(inputs.double())

print(torch.detach(outputs).numpy())
