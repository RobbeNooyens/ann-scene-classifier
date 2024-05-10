import torch
import torch.nn as nn
from model_manager import ModelManager
from config import Config
from tqdm import tqdm


class SceneClassifier(ModelManager):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__(model_name, pretrained)
        self.num_classes = num_classes
        self._modify_classifier()

    def _modify_classifier(self):
        # Adjust the classifier part of the model for scene classification
        if isinstance(self.model.classifier, nn.Sequential):
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, self.num_classes)
        else:
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.num_classes)

    def fine_tune_classifier(self, train_loader, valid_loader, epochs=10):
        """
        Fine-tune only the classifier part of the model on the scene classification task.
        """
        self.model.train()  # Set the model to training mode
        # Freeze all layers except the classifier
        for param in self.model.features.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

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
            print(f'Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader)}')

            # Validate the model
            self.evaluate_model(valid_loader)

    def evaluate_model(self, test_loader):
        """
        Evaluate the model's performance on the test loader and return the accuracy.
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
