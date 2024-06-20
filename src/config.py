from dataclasses import dataclass
from typing import Callable

from torch import nn
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Adam, RMSprop
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
    TRAIN_SPLIT: float = 0.80
    VALID_SPLIT: float = 0.10
    TEST_SPLIT: float = 0.10
    SHUFFLE_DATASET: bool = True
    RANDOM_SEED: int = 0
    DATA_TRANSFORMER: Callable = None

    # Model training
    BASE_MODEL: Module = efficientnet_b0
    BASE_WEIGHTS: WeightsEnum = EfficientNet_B0_Weights.IMAGENET1K_V1
    TRAINABLE_LAYERS: list = None
    OPTIMIZER: Callable = Adam
    CRITERION: Callable = CrossEntropyLoss
    CHECKPOINT: str = None
    USE_SCHEDULER: bool = False

    EARLY_STOPPING_PATIENCE: int = 2
    EARLY_STOPPING_DELTA: float = 0.01
    MIN_EPOCHS: int = 5
    MAX_EPOCHS: int = 40

    LEARNING_RATE: float = 0.0001
    WEIGHT_DECAY: float = 0.0
    CLASSES: list = None

    CLASSIFIER: Module = None
    CHECKPOINT_CLASSIFIER: Module = None


FINETUNE_CONFIG = Configuration(
    MODEL_NAME="finetuned",
    CLASSES=classifier_classes,
    TRAINABLE_LAYERS=["features", "classifier"],
    MIN_EPOCHS=10,
    MAX_EPOCHS=50,
    EARLY_STOPPING_DELTA=0.001,
    EARLY_STOPPING_PATIENCE=5,
    LEARNING_RATE=0.0001,
    CLASSIFIER=nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=1280, out_features=1024, bias=True),
        nn.ReLU(inplace=False),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=1024, out_features=512, bias=True),
        nn.ReLU(inplace=False),
        nn.Linear(in_features=512, out_features=len(classifier_classes), bias=True)
    )
)

GAUSSIAN_BLUR_CONFIG = Configuration(
    MODEL_NAME="gaussian_blur",
    CLASSES=[5, 9, 13, 17, 21],
    MAX_EPOCHS=15,
    DATA_TRANSFORMER=gaussian_blur,
    LEARNING_RATE=0.0001,
    TRAINABLE_LAYERS=["features", "classifier"],
    CHECKPOINT="archive/Gaussian blur pretext/gaussian_blur_pretext.ckpt",
    CLASSIFIER=nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=5, bias=True)
    )
)
GAUSSIAN_BLUR_SCENES_CONFIG = Configuration(
    MODEL_NAME="gaussian_blur_scenes",
    CLASSES=classifier_classes,
    MAX_EPOCHS=15,
    TRAINABLE_LAYERS=["classifier"],
    CHECKPOINT="archive/Gaussian blur pretext/gaussian_blur_pretext.ckpt",
    CLASSIFIER=nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=1280, out_features=1024, bias=True),
        nn.ReLU(inplace=False),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=1024, out_features=512, bias=True),
        nn.ReLU(inplace=False),
        nn.Linear(in_features=512, out_features=len(classifier_classes), bias=True)
    ),
    CHECKPOINT_CLASSIFIER=nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=5, bias=True)
    )
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


# ======================================================================================================================

FINETUNE_CONFIG_ADAM_OPTIMIZER = Configuration(
    MODEL_NAME="finetuned",
    CLASSES=classifier_classes,
    CHECKPOINT="archive/Finetune Adam/finetuned.pth",
    # TRAINABLE_LAYERS=["features", "classifier"],
    # MIN_EPOCHS=10,
    # MAX_EPOCHS=30,
    # EARLY_STOPPING_DELTA=0.001,
    # EARLY_STOPPING_PATIENCE=5,
    # LEARNING_RATE=0.00002,
    # BATCH_SIZE=16,
    # CLASSIFIER=nn.Sequential(
    #     # nn.Dropout(p=0.5, inplace=False),
    #     # nn.Linear(in_features=1280, out_features=640, bias=True),
    #     # nn.ReLU(inplace=False),
    #     nn.Dropout(p=0.3, inplace=False),
    #     nn.Linear(in_features=1280, out_features=len(classifier_classes), bias=True)
    # )
)

FINETUNE_CONFIG_1_HIDDEN_LAYER = Configuration(
    MODEL_NAME="finetuned_1_hidden_layer",
    CLASSES=classifier_classes,
    CHECKPOINT="archive/Finetune 1 hidden layer/finetuned.pth",
    # TRAINABLE_LAYERS=["features", "classifier"],
    # MIN_EPOCHS=10,
    # MAX_EPOCHS=30,
    # EARLY_STOPPING_DELTA=0.001,
    # EARLY_STOPPING_PATIENCE=5,
    # LEARNING_RATE=0.00002,
    # BATCH_SIZE=16,
    # CLASSIFIER=nn.Sequential(
    #     nn.Dropout(p=0.3, inplace=False),
    #     nn.Linear(in_features=1280, out_features=320, bias=True),
    #     nn.ReLU(inplace=False),
    #     # nn.Dropout(p=0.3, inplace=False),
    #     nn.Linear(in_features=320, out_features=len(classifier_classes), bias=True)
    # )
)

FINETUNE_CONFIG_2_HIDDEN_LAYERS = Configuration(
    MODEL_NAME="finetuned_2_hidden_layers",
    CLASSES=classifier_classes,
    CHECKPOINT="archive/Finetune 2 hidden layers/finetuned.pth",
    # TRAINABLE_LAYERS=["features", "classifier"],
    # MIN_EPOCHS=10,
    # MAX_EPOCHS=50,
    # EARLY_STOPPING_DELTA=0.001,
    # EARLY_STOPPING_PATIENCE=5,
    # LEARNING_RATE=0.0001,
    # CLASSIFIER=nn.Sequential(
    #     nn.Dropout(p=0.5, inplace=False),
    #     nn.Linear(in_features=1280, out_features=1024, bias=True),
    #     nn.ReLU(inplace=False),
    #     nn.Dropout(p=0.5, inplace=False),
    #     nn.Linear(in_features=1024, out_features=512, bias=True),
    #     nn.ReLU(inplace=False),
    #     nn.Linear(in_features=512, out_features=len(classifier_classes), bias=True)
    # )
)
FINETUNE_CONFIG_ORIGINAL = Configuration(
    MODEL_NAME="finetuned_original",
    CLASSES=classifier_classes,
    CHECKPOINT="archive/Finetune Original/finetuned.pth",
    # TRAINABLE_LAYERS=["features", "classifier"],
    # MIN_EPOCHS=10,
    # MAX_EPOCHS=50,
    # EARLY_STOPPING_DELTA=0.001,
    # EARLY_STOPPING_PATIENCE=5,
    # BATCH_SIZE=32,
    # # Parameters from paper
    # OPTIMIZER=RMSprop,
    # WEIGHT_DECAY=1e-5,
    # LEARNING_RATE=0.001,
    # CLASSIFIER=nn.Sequential(
    #     nn.Dropout(p=0.2, inplace=True),
    #     nn.Linear(in_features=1280, out_features=len(classifier_classes), bias=True)
    # ),
    # USE_SCHEDULER=True,
)