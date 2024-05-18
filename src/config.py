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
    EPOCHS = 1 # TODO Vergroten
    LEARNING_RATE = 0.001

    # Hardware configuration
    USE_GPU = True

    # Pretext tasks configurations (if applicable)
    PRETEXT_TASKS = {
        'gaussian_blur': {
            'model_name': 'EfficientNet-B0',  # Model name to use for the pretext task
            'classes': [5, 9, 13, 17, 21],  # Example for Gaussian blur sizes
            'model_path': 'models/gaussian_blur.pth',
            'checkpoint_path': 'models/gaussian_blur_checkpoint.pth',
            'scene_model_path': 'models/scene_gaussian_blur.pth',
            'scene_checkpoint_path': 'models/scene_gaussian_blur_checkpoint.pth'
        },
        'black_white_perturbation': {
            'model_name': 'EfficientNet-B0',  # Model name to use for the pretext task
            'classes': [0, 1],  # Black or white perturbation
            'model_path': 'models/perturbation.pth',
            'checkpoint_path': 'models/perturbation_checkpoint.pth',
            'scene_model_path': 'models/scene_perturbation.pth',
            'scene_checkpoint_path': 'models/scene_perturbation_checkpoint.pth'
        }
    }

    # Adjust this according to the number of classes in your dataset if known
    NUM_CLASSES = 15  # Assuming there are 15 different scenes as per the dataset description
