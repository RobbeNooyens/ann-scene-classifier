import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchvision import models
from torchvision.models.efficientnet import EfficientNet_B0_Weights

from config import Config


class ModelManager:
    def __init__(self, model_name, pretrained=True, use_checkpoint=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = self.load_model(model_name, pretrained, use_checkpoint).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

    def load_model(self, model_name, pretrained, use_checkpoint):
        """
        Load a model from a checkpoint if specified and exists, otherwise load from torchvision.
        """
        checkpoint_path = f'{Config.CHECKPOINT_SAVE_PATH}'
        if use_checkpoint and os.path.exists(checkpoint_path):
            model = models.efficientnet_b0()  # Create an instance of the model first
            model.load_state_dict(torch.load(checkpoint_path))  # Load the state dictionary
            print(f"Loaded checkpoint model from {checkpoint_path}.")
        elif pretrained:
            if model_name == 'EfficientNet-B0':
                weights = EfficientNet_B0_Weights.DEFAULT
                model = models.efficientnet_b0(weights=weights)
                print("Loaded pre-trained EfficientNet-B0.")
            else:
                raise ValueError(f"Model {model_name} not supported.")
        else:
            print("Initializing a new model as no checkpoint found and pretrained is set to False.")
            model = models.efficientnet_b0()  # Example: Creating a new instance without pre-trained weights

        # Make all parameters trainable
        if model:
            for param in model.parameters():
                param.requires_grad = True

        return model

    def train_model(self, train_loader, valid_loader, epochs=10):
        """
        Train the model using the specified training and validation loaders for a number of epochs,
        including validation after each epoch and plot the training and validation loss and accuracy.
        """
        train_losses = []
        valid_losses = []
        valid_accuracies = []

        self.model.train()
        for epoch in range(epochs):
            total_train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            # Calculate average losses
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            valid_accuracy, avg_valid_loss = self.evaluate_model(valid_loader)
            valid_losses.append(avg_valid_loss)
            valid_accuracies.append(valid_accuracy)

            # Save the model after each epoch
            self.save_model(f'{Config.CHECKPOINT_SAVE_PATH}')

            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%')

        # Plotting
        self.plot_training_results(train_losses, valid_losses, valid_accuracies)

    def plot_training_results(self, train_losses, valid_losses, valid_accuracies):
        """
        Plot the training and validation losses and validation accuracy.
        """
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, valid_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, valid_accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.savefig(f"plots/training_plot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")

    def evaluate_model(self, test_loader):
        """
        Evaluate the model's performance on the test loader and return the accuracy and loss.
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        average_loss = total_loss / len(test_loader)
        return accuracy, average_loss

    def save_model(self, path):
        """
        Save the trained model to the specified path.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
