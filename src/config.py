class Config:
    # Paths
    DATA_PATH = "15-Scene"
    MODEL_SAVE_PATH = "models/EfficientNet-B0.pth"
    CHECKPOINT_SAVE_PATH = "models/EfficientNet-B0-checkpoint.pth"

    # Data loading
    BATCH_SIZE = 64
    TRAINING_SPLIT = 0.01
    VALIDATION_SPLIT = 0.01
    SHUFFLE_DATASET = True
    RANDOM_SEED = 42

    # Model training
    EPOCHS = 2 # TODO Vergroten
    LEARNING_RATE = 0.001

    # Hardware configuration
    USE_GPU = True

    # Pretext tasks configurations (if applicable)
    PRETEXT_TASKS = {
        'gaussian_blur': {
            'classes': 5,  # Example for Gaussian blur sizes
            'pretext_model_path': 'models/EfficientNet-B0_pretext_gaussian_blur.pth'
        },
        'black_white_perturbation': {
            'classes': 2,  # Black or white perturbation
            'pretext_model_path': 'models/EfficientNet-B0_pretext_bw_perturbation.pth'
        }
    }

    # Adjust this according to the number of classes in your dataset if known
    NUM_CLASSES = 15  # Assuming there are 15 different scenes as per the dataset description
