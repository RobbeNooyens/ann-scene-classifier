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
    def __init__(self, pretext_task, checkpoint=None):
        config = Config.PRETEXT_TASKS[pretext_task]
        self.model_path = config['model_path']
        self.checkpoint_path = config['checkpoint_path']
        super().__init__(config['model_name'], checkpoint=checkpoint)
        self.pretext_task = pretext_task
        self.classes = config['classes']
        self.config = config
        self.original_classifier = None  # To store the original classifier
        self.image_transformer = ImageTransformer()
        self.modify_classifier_for_pretext()

    def modify_classifier_for_pretext(self):
        """
        Modify the classifier for the specific pretext task and save the original classifier.
        """
        # Load correct paths
        self.model_path = self.config['model_path']
        self.checkpoint_path = self.config['checkpoint_path']

        # Save the original classifier
        if hasattr(self.model, 'classifier') and isinstance(self.model.classifier, nn.Sequential):
            self.original_classifier = self.model.classifier[-1]
        # Assume the last layer is the classifier which we need to modify
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_ftrs, len(self.classes))


    def prepare_for_scene_classification(self):
        """
        Prepare the model for scene classification by restoring the original classifier
        and freezing the feature extraction layers.
        """
        # Load correct paths
        self.model_path = self.config['scene_model_path']
        self.checkpoint_path = self.config['scene_checkpoint_path']

        # Freeze all feature extraction layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Restore the original classifier
        if self.original_classifier is not None:
            self.model.classifier[-1] = self.original_classifier

    def modify_images(self, images, labels):
        data = []
        if self.pretext_task == 'gaussian_blur':
            for i in range(len(self.classes)):
                data.append((
                    torch.stack([self.image_transformer.gaussian_blur(img, kernel_size=self.classes[i]) for img in images]),
                    torch.tensor([i] * images.size(0))
                ))
        elif self.pretext_task == 'black_white_perturbation':
            for i in range(len(self.classes)):
                data.append((
                    torch.stack([self.image_transformer.perturbation(img, color=self.classes[i]) for img in images]),
                    torch.tensor([i] * images.size(0))
                ))
        return data

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

    def get_model_path(self):
        return self.model_path

    def get_checkpoint_path(self):
        return self.checkpoint_path


