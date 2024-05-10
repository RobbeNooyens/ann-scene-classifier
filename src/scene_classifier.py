import torch
import torch.nn as nn
from model_manager import ModelManager
from config import Config


class SceneClassifier(ModelManager):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__(model_name, pretrained)
        # Replace the classifier with a new one for scene classification
        self.num_classes = num_classes
        self._modify_classifier()

    def _modify_classifier(self):
        """
        Modify the classifier part of the model for scene classification.
        """
        # Assuming the model has an attribute `classifier` that we want to replace
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, self.num_classes).to(self.device)

    def train_model(self, train_loader, optimizer=None, criterion=None, epochs=10):
        """
        Train the classifier of the model on the scene classification task.
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=Config.LEARNING_RATE)
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')

    def evaluate_model(self, test_loader):
        """
        Evaluate the model on the scene classification task.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the scene classification task: {accuracy}%')
        return accuracy
