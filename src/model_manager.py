import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
from torchvision.models.efficientnet import EfficientNet_B0_Weights

from config import Config


class ModelManager:
    def __init__(self, model_name, pretrained=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_name, pretrained).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

    def load_model(self, model_name, pretrained):
        """
        Load a pre-trained model or initialize a new one, making all layers trainable.
        """
        if model_name == 'EfficientNet-B0' and pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
            # Set all model parameters to be trainable
            for param in model.parameters():
                param.requires_grad = True
            print("Loaded pre-trained EfficientNet-B0 with all layers trainable.")
        else:
            model = None
        return model

    def train_model(self, train_loader, epochs=10):
        """
        Train the model using the specified training loader and number of epochs.
        """
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for images, labels in tqdm(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            average_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}, Loss: {average_loss}')

    def evaluate_model(self, test_loader):
        """
        Evaluate the model's performance on the test loader and return the accuracy.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy of the model: {accuracy:2f}%')
        return accuracy

    def save_model(self, path):
        """
        Save the trained model to the specified path.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
