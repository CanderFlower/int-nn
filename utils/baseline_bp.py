import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import read_image_file, read_label_file
from time import time

# Hyperparameters
dim_input = 28 * 28
dim1 = 100
dim2 = 50
num_classes = 10
epochs = 20
batch_size = 20
lr = 1.0 / 100

# Data
class CustomMNIST(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, transform=None):
        self.images = read_image_file(image_path)
        self.labels = read_label_file(label_path)
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index].float().div(255).view(-1)  # flatten
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)

trainset = CustomMNIST(
    image_path="../dataset/mnist/train-images.idx3-ubyte",
    label_path="../dataset/mnist/train-labels.idx1-ubyte"
)
testset = CustomMNIST(
    image_path="../dataset/mnist/t10k-images.idx3-ubyte",
    label_path="../dataset/mnist/t10k-labels.idx1-ubyte"
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Model
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(dim_input, dim1)
        self.fc2 = nn.Linear(dim1, dim2)
        self.fc3 = nn.Linear(dim2, num_classes)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(device)
model = DNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# One-hot encoding helper
def one_hot(labels, num_classes):
    return torch.eye(num_classes, device=labels.device)[labels]

# Training
print("Epoch,\tTrainLoss,\tTrainAcc,\tTestAcc")
start = time()
for ep in range(1, epochs + 1):
    model.train()
    total_loss, total_correct = 0, 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        targets = one_hot(labels, num_classes).to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()

    # Test
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_correct += (outputs.argmax(dim=1) == labels).sum().item()

    print(f"{ep},\t{int(total_loss)},\t{total_correct/len(trainset)*100:.2f}%,\t\t{test_correct/len(testset)*100:.2f}%")

end = time()
print(f"Training time: {end - start:.2f} seconds")
