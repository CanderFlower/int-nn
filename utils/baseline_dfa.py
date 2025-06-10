import torch
import torch.nn as nn
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
class DNN_DFA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(dim_input, dim1)
        self.fc2 = nn.Linear(dim1, dim2)
        self.fc3 = nn.Linear(dim2, num_classes)
        self.tanh = nn.Tanh()

        # Fixed random feedback matrices (for DFA)
        self.B1 = torch.randn(num_classes, dim1, device=device) * (1.0 / (dim1 ** 0.5))
        self.B2 = torch.randn(num_classes, dim2, device=device) * (1.0 / (dim2 ** 0.5))

    def forward(self, x):
        self.x1 = self.tanh(self.fc1(x))  # save for backward
        self.x2 = self.tanh(self.fc2(self.x1))
        self.out = self.tanh(self.fc3(self.x2))
        return self.out

    def dfa_backward(self, targets, lr):
        # Calculate output error (MSE loss gradient)
        delta_out = 2 * (self.out - targets) / targets.size(0)  # batch averaged

        # Feedback to hidden layers using fixed matrices
        delta_fc3 = delta_out * (1 - self.out ** 2)

        delta_fc2 = (delta_out @ self.B2) * (1 - self.x2 ** 2)
        delta_fc1 = (delta_out @ self.B1) * (1 - self.x1 ** 2)

        # Update weights manually
        self.fc3.weight.data -= lr * delta_fc3.t() @ self.x2
        self.fc3.bias.data -= lr * delta_fc3.sum(dim=0)

        self.fc2.weight.data -= lr * delta_fc2.t() @ self.x1
        self.fc2.bias.data -= lr * delta_fc2.sum(dim=0)

        # Input layer update requires input x
        # Assuming inputs saved as self.input (set in training loop)
        self.fc1.weight.data -= lr * delta_fc1.t() @ self.input
        self.fc1.bias.data -= lr * delta_fc1.sum(dim=0)

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(device)
model = DNN_DFA().to(device)
criterion = nn.MSELoss()

# One-hot encoding helper
def one_hot(labels, num_classes):
    return torch.eye(num_classes, device=labels.device)[labels]

# Training
print("Epoch,\tTrainLoss,\tTrainAcc,\tTestAcc")
start = time()
for ep in range(1, epochs+1):
    model.train()
    total_loss, total_correct = 0, 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        targets = one_hot(labels, num_classes).to(device)
        
        model.input = inputs  # save input for dfa_backward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()

        model.dfa_backward(targets, lr)
    
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
