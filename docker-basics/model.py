import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


class TrainModels:
    def __init__(self, epochs, learning_rate):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=0.9
        )
        self.epochs = epochs

    def train(self, training_dataloader):
        training_log = []
        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(training_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if (i + 1) % 100 == 0:
                    print(
                        "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                            epoch + 1,
                            self.epochs,
                            i + 1,
                            len(training_dataloader),
                            loss.item(),
                        )
                    )

    def test(self, testing_dataloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testing_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(
            "Accuracy of the network on the 10000 test images: {} %".format(
                100 * correct / total
            )
        )

    def save_model(self, model_name):
        torch.save(self.model.state_dict(), model_name)
