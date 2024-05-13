import os

from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision.models import EfficientNet_B0_Weights
from torchvision.transforms import functional as F


from config import Config

transforms = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

class DataHandler:
    def __init__(self):
        self.transform = transforms

    def load_data(self):
        dataset = datasets.ImageFolder(Config.DATA_PATH, transform=self.transform)
        train_dataset, valid_dataset, _ = random_split(dataset, [Config.TRAINING_SPLIT, Config.VALIDATION_SPLIT, 1-Config.TRAINING_SPLIT-Config.VALIDATION_SPLIT])

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=Config.SHUFFLE_DATASET,
                                  num_workers=4, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4,
                                  pin_memory=True)

        return train_loader, valid_loader

    def load_image(self, image_path):
        """
        Loads a single image and applies the pre-configured transformations.
        """
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        self.save_transformed_image(image, 'original_transformed.jpg')
        return image

    def save_transformed_image(self, image_tensor, save_path):
        """
        Saves a transformed tensor image to a file.
        """
        pil_image = F.to_pil_image(image_tensor)
        pil_image.save(save_path)

    def get_transform(self):
        """
        Returns the current transformation pipeline.
        """
        return self.transform
