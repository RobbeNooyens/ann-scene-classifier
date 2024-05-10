import random

import torch
import torch.nn as nn
from model_manager import ModelManager
from torchvision.transforms import functional as F
from tqdm import tqdm

from src.config import Config


class PretextTask(ModelManager):
    def __init__(self, model_name, pretext_task, num_classes, pretrained=True):
        super().__init__(model_name, pretrained)
        self.pretext_task = pretext_task
        self.num_classes = num_classes
        self.original_classifier = None  # To store the original classifier
        self.modify_classifier_for_pretext()

    def modify_classifier_for_pretext(self):
        """
        Modify the classifier for the specific pretext task and save the original classifier.
        """
        # Save the original classifier
        if hasattr(self.model, 'classifier') and isinstance(self.model.classifier, nn.Sequential):
            self.original_classifier = self.model.classifier[-1]
        # Assume the last layer is the classifier which we need to modify
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_ftrs, self.num_classes)

    def restore_original_classifier(self):
        """
        Restore the original classifier that was saved before the pretext task modification.
        """
        if self.original_classifier is not None:
            self.model.classifier[-1] = self.original_classifier

    def prepare_for_scene_classification(self):
        """
        Prepare the model for scene classification by restoring the original classifier
        and freezing the feature extraction layers.
        """
        # Freeze all feature extraction layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Restore the original classifier
        self.restore_original_classifier()

    def train_pretext_task(self, train_loader, optimizer, criterion, epochs):
        """
        Train the model on the pretext task.
        """
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in tqdm(train_loader):
                if self.pretext_task == 'gaussian_blur':
                    kernels = [5, 9, 13, 17, 21]  # Example kernel sizes
                    kernel = random.choice(kernels)
                    labels = torch.tensor([kernels.index(kernel)] * images.size(0))
                    images = torch.stack([F.gaussian_blur(img, kernel_size=kernel) for img in images])
                elif self.pretext_task == 'black_white_perturbation':
                    sizes = [0, 1]  # 0 for black, 1 for white
                    perturbation = random.choice(sizes)
                    labels = torch.tensor([perturbation] * images.size(0))
                    images = torch.stack([self.apply_black_white_perturbation(img, perturbation) for img in images])

                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

    def apply_black_white_perturbation(self, img, perturbation):
        """
        Apply a black or white perturbation to a random region of the image.
        """
        start_x, start_y = random.randint(0, img.shape[1] - 10), random.randint(0, img.shape[2] - 10)
        end_x, end_y = start_x + 10, start_y + 10
        perturbed_img = img.clone()
        perturbed_img[:, start_x:end_x, start_y:end_y] = 0 if perturbation == 0 else 1
        return perturbed_img

    def fine_tune_classifier(self, train_loader, valid_loader, epochs):
        """
        Fine-tunes the classifier part of the model on the scene classification task.
        """
        self.model.train()
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

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')
            self.evaluate_model(valid_loader)

    def evaluate_model(self, valid_loader):
        """
        Evaluate the fine-tuned model on the validation set for scene classification.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy}%')
        return accuracy

