import random

import torch
import torch.nn as nn
from model_manager import ModelManager
from torchvision.transforms import functional as F
from tqdm import tqdm

class PretextTask(ModelManager):
    def __init__(self, model_name, pretext_task, num_classes, pretrained=True):
        super().__init__(model_name, pretrained)
        self.pretext_task = pretext_task
        self.num_classes = num_classes
        self.modify_classifier()  # Modify the classifier to fit the pretext task needs

    def modify_classifier(self):
        """
        Adjust the classifier part of the model for the specific pretext task.
        This assumes the classifier ends with a Linear layer we want to replace.
        """
        # Accessing the last layer of the classifier which is a linear layer in EfficientNet
        if isinstance(self.model.classifier, nn.Sequential):
            # Replace the last layer
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, self.num_classes)
        else:
            # Direct replacement if it's a single Linear layer
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.num_classes)

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

    # def generate_labels(self, images):
    #     """
    #     Generate labels according to the specific pretext task being run.
    #     """
    #     if self.pretext_task == 'gaussian_blur':
    #         # Implementation to generate labels for Gaussian blur sizes
    #         pass
    #     elif self.pretext_task == 'black_white_perturbation':
    #         # Implementation to generate labels for black and white perturbation
    #         pass
    #     return torch.randint(0, self.num_classes, (images.size(0),))

