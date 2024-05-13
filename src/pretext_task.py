import random

import torch
import torch.nn as nn
from PIL import Image

from model_manager import ModelManager
from torchvision.transforms import functional as F
from tqdm import tqdm

from src.config import Config
from src.transformations import ImageTransformer


class PretextTask(ModelManager):
    def __init__(self, pretext_task):
        config = Config.PRETEXT_TASKS[pretext_task]
        super().__init__(config['model_name'], True)
        self.pretext_task = pretext_task
        self.classes = config['classes']
        self.original_classifier = None  # To store the original classifier
        self.image_transformer = ImageTransformer()
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
        self.model.classifier[-1] = nn.Linear(num_ftrs, len(self.classes))

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

    def modify_images(self, images, labels):
        true_label = random.randint(0, len(self.classes) - 1)
        labels = torch.tensor([true_label] * images.size(0))
        if self.pretext_task == 'gaussian_blur':
            images = torch.stack(
                [self.image_transformer.gaussian_blur(img, kernel_size=self.classes[true_label]) for img in images])
        elif self.pretext_task == 'black_white_perturbation':
            images = torch.stack(
                [self.image_transformer.perturbation(img, color=self.classes[true_label]) for img in images])
        return images, labels

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

    def classify_image(self, image_tensor):
        """
        Receive a pre-processed image tensor, apply the pretext task transformation, classify it,
        and optionally save the transformed image.
        """
        true_label = random.randint(0, len(self.classes) - 1)
        if self.pretext_task == 'gaussian_blur':
            transformed_image = self.image_transformer.gaussian_blur(image_tensor, kernel_size=self.classes[true_label])
        elif self.pretext_task == 'black_white_perturbation':
            transformed_image = self.image_transformer.perturbation(image_tensor, color=self.classes[true_label])
        else:
            raise ValueError("Unknown pretext task.")

        transformed_image = transformed_image.unsqueeze(0)


        # Classify the transformed image
        self.model.eval()
        transformed_image = transformed_image.to(self.device)
        with torch.no_grad():
            outputs = self.model(transformed_image)
            _, predicted = outputs.max(1)
        transformed_image = transformed_image.squeeze(0)

        return transformed_image, predicted.item(), true_label

