import torch
import numpy as np
import pandas as pd


def one(glob_input):
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

        # if epoch % 10 == 0:
        # print('Epoch:', epoch, 'Loss:', loss.item())

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model2.pt')

    """this is a new try 1"""
    # Load the trained model
    # model = torch.load('trained_model1.pt')

    # Get the input data from the terminal
    input_str = glob_input
    input_data = np.array([np.double(x) for x in input_str.split()])
    # Convert the input data to a tensor
    inputs = torch.from_numpy(input_data)
    # Get the output of the model
    model = model.double()
    outputs = model(inputs.double())

    # Print the output data to the terminal

    x = torch.detach(outputs).numpy()
    x = list(x)
    out = [0, 0, 0]
    i = 0
    if max(x) > 0.7:
        i = x.index(max(x))
        out[i] = 1
    next_out = out
    next_input = glob_input + " " + str(next_out[0]) + " " + str(next_out[1]) + " " + str(next_out[2])
    return next_out, next_input


def two(prev_out):
    input_data = pd.read_csv('csvs/newinput_with_3outputs.csv')
    input_data = input_data.values.astype(np.float32)

    output_data = pd.read_csv('csvs/bus_number.csv')
    output_data = output_data.values.astype(np.float32)

    # Define the neural network model
    class NeuralNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(63, 255)
            self.fc2 = torch.nn.Linear(255, 47)
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
    for epoch in range(1000):
        inputs = torch.from_numpy(input_data)
        targets = torch.from_numpy(output_data)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if epoch % 10 == 0:
            # print('Epoch:', epoch, 'Loss:', loss.item())

    torch.save(model.state_dict(), 'model.pt')
    # input_str = input("Enter the input data (separated by spaces): ")
    input_str = prev_out
    input_data = np.array([np.double(x) for x in input_str.split()])

    # Convert the input data to a tensor
    inputs = torch.from_numpy(input_data)

    # Get the output of the model
    model = model.double()
    outputs = model(inputs.double())

    x = torch.detach(outputs).numpy()
    y = []
    for i in x:
        y.append(float(i))
    next_out2 = y.index(max(y))
    return next_out2

