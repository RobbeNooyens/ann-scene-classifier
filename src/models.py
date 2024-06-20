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
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.config = config
        self.load()
        self.freeze_layers()
        self.criterion = self.config.CRITERION()
        self.optimizer = self.config.OPTIMIZER(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200) if config.USE_SCHEDULER else None
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
            self.model.classifier = self.config.CHECKPOINT_CLASSIFIER if self.config.CHECKPOINT_CLASSIFIER else self.config.CLASSIFIER
            self.model.load_state_dict(torch.load(checkpoint))
            if self.config.CHECKPOINT_CLASSIFIER:
                # Replace classifier if checkpoint classifier differs from classifier
                self.model.classifier = self.config.CLASSIFIER
            print(f"Loaded checkpoint model from {checkpoint}.")
        elif base_weights:
            self.model = base_model(weights=base_weights)
            self.model.classifier = self.config.CLASSIFIER
            print("Loaded model with predefined weights.")
        else:
            self.model = base_model()
            self.model.classifier = self.config.CLASSIFIER
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

    def early_stopping(self, val_loss, best_loss, counter):
        if best_loss is None or val_loss < best_loss - self.config.EARLY_STOPPING_DELTA:
            best_loss = val_loss
            counter = 0
            return False, best_loss, counter
        counter += 1
        if counter >= self.config.EARLY_STOPPING_PATIENCE:
            return True, best_loss, counter
        else:
            return False, best_loss, counter

    def train_model_pre_load(self, train_loader, valid_loader):
        """
        Train the model using the specified training and validation loaders for a number of epochs,
        including validation after each epoch and plot the training and validation loss and accuracy.
        """
        def pre_load_data(data_loader):
            images_list, labels_list = [], []
            for images, labels in data_loader:
                images_list.append(images.to(self.device))
                labels_list.append(labels.to(self.device))
            return images_list, labels_list

        train_images, train_labels = pre_load_data(train_loader)
        valid_images, valid_labels = pre_load_data(valid_loader)
        self.train_model(list(zip(train_images, train_labels)), list(zip(valid_images, valid_labels)))

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

        self.model.to(self.device)

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
            if self.scheduler:
                self.scheduler.step()

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
            print(f'Epoch {epoch + 1}, Total Train Loss: {total_train_loss:.4f}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%')

            # Early stopping
            stop, best_loss, counter = self.early_stopping(avg_valid_loss, best_loss, epochs_no_improve)
            if stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Plotting
        self.plot_training_results(train_losses, valid_losses, valid_accuracies)
        self.save(self.config.MODEL_FOLDER, f"{self.config.MODEL_NAME}.pth")

        # Clean up
        self.model.to("cpu")
        torch.mps.empty_cache()

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

        if not os.path.exists("plots"):
            os.makedirs("plots")
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
        self.model.to(self.device)
        accuracy, loss = self.evaluate(data_loader)
        self.logger.log(self.config.MODEL_NAME, "test", loss=loss, accuracy=accuracy)
        self.model.to("cpu")

    def save(self, folder, name):
        """
        Save the trained model to the specified path.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, name)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
