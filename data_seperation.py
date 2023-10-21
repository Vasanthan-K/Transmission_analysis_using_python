import csv
import random

csv_file_path = 'your_file.csv'
data = []
with open("Book3.csv", 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        data.append(row)

required_data = data[6:]
i = 0
training = []
testing = []
training_num = []
while i < 525:
    rand = random.randint(0, 695)
    if rand not in training_num:
        training_num.append(rand)
        training.append(required_data[rand])
        i += 1

print(training)
for i in range(len(required_data)):
    if i not in training_num:
        testing.append(required_data[i])

print("THE TESTING DAT STARTS HERE\n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \n ")
print(testing)

with open('training_data.csv', "x") as file:
    writer = csv.writer(file)
    writer.writerows(training)

with open('testing_data.csv', "x") as file:
    writer = csv.writer(file)
    writer.writerows(testing)

