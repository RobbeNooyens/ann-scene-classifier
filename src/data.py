from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.models import EfficientNet_B0_Weights
from torchvision.transforms.functional import to_pil_image

from config import Configuration

transforms = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()


class TransformedDataset(Dataset):
    def __init__(self, dataset, labels, tranform_func):
        self.dataset = dataset
        self.labels = labels
        self.transform_func = tranform_func

    def __len__(self):
        return len(self.dataset) * len(self.labels)

    def __getitem__(self, idx):
        image_idx = idx // len(self.labels)
        transform_idx = idx % len(self.labels)
        image, _ = self.dataset[image_idx]
        transformed_image = self.transform_func(image, transform_idx)
        return transformed_image, transform_idx


class DataHandler:
    def __init__(self, config: Configuration):
        self.transform = transforms
        self.config = config

    def load(self) -> (DataLoader, DataLoader, DataLoader):
        dataset = datasets.ImageFolder(self.config.DATA_PATH, transform=self.transform)
        train, valid, test = random_split(dataset, [self.config.TRAIN_SPLIT, self.config.VALID_SPLIT, self.config.TEST_SPLIT])

        if self.config.DATA_TRANSFORMER is not None:
            labels = list(range(len(self.config.CLASSES)))
            train = TransformedDataset(train, labels, self.config.DATA_TRANSFORMER)
            valid = TransformedDataset(valid, labels, self.config.DATA_TRANSFORMER)
            test = TransformedDataset(test, labels, self.config.DATA_TRANSFORMER)

        train_loader = DataLoader(train, batch_size=self.config.BATCH_SIZE, shuffle=self.config.SHUFFLE_DATASET, num_workers=4)
        valid_loader = DataLoader(valid, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader = DataLoader(test, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=4)

        return train_loader, valid_loader, test_loader

    def convert_image(self, input_path: str, output_path: str, label: int):
        """
        Loads a single image and applies the pre-configured transformations.
        """
        image = Image.open(input_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.config.DATA_TRANSFORMER:
            image = self.config.DATA_TRANSFORMER(image, label)
        pil_image = to_pil_image(image)
        pil_image.save(output_path)

