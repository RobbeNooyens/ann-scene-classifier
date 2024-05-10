from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision.models import EfficientNet_B0_Weights

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
