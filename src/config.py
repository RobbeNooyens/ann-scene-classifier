from dataclasses import dataclass
from typing import Callable

from torch.nn import Module
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, WeightsEnum

from transformations import gaussian_blur, perturbation

classifier_classes = ["bedroom", "suburb", "industrial", "kitchen", "living room", "coast", "forest", "highway",
                      "inside cite", "mountain", "open country", "street", "tall building", "office", "store"]


@dataclass
class Configuration:
    # Filesystem
    MODEL_NAME: str
    DATA_PATH: str = "15-Scene"
    MODEL_FOLDER: str = "models"
    CHECKPOINT_FOLDER: str = "checkpoints"

    # Data loading
    BATCH_SIZE: int = 32
    TRAIN_SPLIT: float = 0.01
    VALID_SPLIT: float = 0.01
    TEST_SPLIT: float = 0.98
    SHUFFLE_DATASET: bool = True
    RANDOM_SEED: int = 42
    DATA_TRANSFORMER: Callable = None

    # Model training
    BASE_MODEL: Module = efficientnet_b0
    BASE_WEIGHTS: WeightsEnum = EfficientNet_B0_Weights.IMAGENET1K_V1
    TRAINABLE_LAYERS: list = None
    CHECKPOINT: str = None
    EARLY_STOPPING_PATIENCE: int = 5
    MIN_EPOCHS: int = 1
    MAX_EPOCHS: int = 1
    LEARNING_RATE: float = 0.001
    CLASSES: list = None
    CLASSIFIER_DROPOUT: float = 0.2

    # Hardware configuration
    USE_GPU: bool = True


FINETUNE_CONFIG = Configuration(
    MODEL_NAME="finetuned",
    CLASSES=classifier_classes,
    TRAINABLE_LAYERS=["classifier"],
)

GAUSSIAN_BLUR_CONFIG = Configuration(
    MODEL_NAME="gaussian_blur",
    CLASSES=[5, 9, 13, 17, 21],
    DATA_TRANSFORMER=gaussian_blur,
    CHECKPOINT="models/gaussian_blur.pth",
    TRAINABLE_LAYERS=["features", "classifier"],
)
GAUSSIAN_BLUR_SCENES_CONFIG = Configuration(
    MODEL_NAME="gaussian_blur_scenes",
    CLASSES=classifier_classes,
    TRAINABLE_LAYERS=["classifier"],
)

PERTURBATION_CONFIG = Configuration(
    MODEL_NAME="perturbation",
    CLASSES=["black", "white"],
    DATA_TRANSFORMER=perturbation,
    TRAINABLE_LAYERS=["features", "classifier"],
)
PERTURBATION_SCENES_CONFIG = Configuration(
    MODEL_NAME="perturbation_scenes",
    CLASSES=classifier_classes,
    TRAINABLE_LAYERS=["classifier"],
)
