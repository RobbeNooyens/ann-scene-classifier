import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.nn import Sequential
from tqdm import tqdm

from config import Configuration
from logger import Logger


class ModelManager:
    def __init__(self, config: Configuration):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.load()
        self.freeze_layers()
        self.modify_classifier_layer()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.logger = Logger()

    def load(self):
        """
        Load a model from a checkpoint if specified and exists, otherwise load from torchvision.
        """
        base_model = self.config.BASE_MODEL
        base_weights = self.config.BASE_WEIGHTS
        checkpoint = self.config.CHECKPOINT

        if not base_model:
            raise ValueError("Model not specified.")
        elif checkpoint and os.path.exists(checkpoint):
            self.model = base_model()
            self.modify_classifier_layer()
            self.model.load_state_dict(torch.load(checkpoint))
            print(f"Loaded checkpoint model from {checkpoint}.")
        elif base_weights:
            self.model = base_model(weights=base_weights)
            print("Loaded model with predefined weights.")
        else:
            self.model = base_model()
            print("Loaded model without predefined weights.")
        self.model.to(self.device)

    def freeze_layers(self):
        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze specified layers
        for layer_name in self.config.TRAINABLE_LAYERS:
            layer = dict(self.model.named_children()).get(layer_name, None)
            if layer and isinstance(layer, Sequential):
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                print(f"Layer {layer_name} not found in model.")

    def inspect(self):
        for name, layer in self.model.named_children():
            print(f"Layer name: {name} \t Layer type: {type(layer)}")

    def train_model(self, train_loader, valid_loader):
        """
        Train the model using the specified training and validation loaders for a number of epochs,
        including validation after each epoch and plot the training and validation loss and accuracy.
        """
        train_losses = []
        valid_losses = []
        valid_accuracies = []

        # Early stopping
        best_loss = float('inf')
        epochs_no_improve = 0
        patience = self.config.EARLY_STOPPING_PATIENCE

        for epoch in range(self.config.MAX_EPOCHS):
            self.model.train()
            total_train_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False):
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
            valid_accuracy, avg_valid_loss = self.evaluate(valid_loader)
            valid_losses.append(avg_valid_loss)
            valid_accuracies.append(valid_accuracy)

            # Save the model after each epoch
            if self.config.CHECKPOINT_FOLDER:
                self.save(self.config.CHECKPOINT_FOLDER, f"{self.config.MODEL_NAME}.{epoch+1}.ckpt")

            # Log values
            self.logger.log(self.config.MODEL_NAME, "train", epoch=epoch + 1, loss=avg_train_loss)
            self.logger.log(self.config.MODEL_NAME, "valid", epoch=epoch + 1, loss=avg_valid_loss, accuracy=valid_accuracy)

            # Early stopping
            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%')

        # Plotting
        self.plot_training_results(train_losses, valid_losses, valid_accuracies)
        self.save(self.config.MODEL_FOLDER, f"{self.config.MODEL_NAME}.pth")

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

        plt.savefig(f"plots/training_plot_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.png")

    def evaluate(self, data_loader):
        """
        Evaluate the model's performance on the a test loader and return the accuracy and loss.
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        loss = total_loss / len(data_loader)
        return accuracy, loss

    def evaluate_test_set(self, data_loader):
        accuracy, loss = self.evaluate(data_loader)
        self.logger.log(self.config.MODEL_NAME, "test", loss=loss, accuracy=accuracy)

    def modify_classifier_layer(self):
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=self.config.CLASSIFIER_DROPOUT, inplace=True),
            nn.Linear(in_features=1280, out_features=len(self.config.CLASSES), bias=True)
        )

    def save(self, folder, name):
        """
        Save the trained model to the specified path.
        """
        path = os.path.join(folder, name)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


class PretextTask(ModelManager):
    def __init__(self, pretext_config: Configuration, scene_config: Configuration):
        super().__init__(pretext_config)
        self.pretext_config = pretext_config
        self.scene_config = scene_config
        self.current_model = pretext_config.MODEL_NAME

    def switch_model(self, new_config):
        """
        Switch the model configuration and modify the classifier layer.
        """
        if self.current_model != new_config.MODEL_NAME:
            self.config = new_config
            self.modify_classifier_layer()
            self.freeze_layers()
            self.current_model = new_config.MODEL_NAME

    def train_pretext_model(self, train_loader, valid_loader):
        """
        Train the pretext model.
        """
        self.switch_model(self.pretext_config)
        self.train_model(train_loader, valid_loader)

    def train_scene_model(self, train_loader, valid_loader):
        """
        Train the scene classification model.
        """
        self.switch_model(self.scene_config)
        self.train_model(train_loader, valid_loader)