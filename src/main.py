from config import Config
from data_handler import DataHandler
from model_manager import ModelManager
from pretext_task import PretextTask
from scene_classifier import SceneClassifier
from experiment_tracker import ExperimentTracker


def main():
    # Data preparation
    # This step covers the 'DataHandler' implementation as per the assignment.
    data_handler = DataHandler()
    train_loader, valid_loader = data_handler.load_data()

    # Supervised Learning
    # Here we are loading and training a pre-trained model as specified in the assignment.
    print("Training supervised model...")
    supervised_model = ModelManager('EfficientNet-B0', pretrained=True)
    supervised_model.train_model(train_loader, epochs=Config.EPOCHS)
    supervised_accuracy = supervised_model.evaluate_model(valid_loader)
    supervised_model.save_model(Config.MODEL_SAVE_PATH)

    # Self-Supervised Learning: Gaussian Blurring
    # Implementing the first self-supervised learning task: Gaussian Blurring.
    print("Training model on Gaussian Blur pretext task...")
    gaussian_model = PretextTask('EfficientNet-B0', 'gaussian_blur', Config.PRETEXT_TASKS['gaussian_blur']['classes'])
    gaussian_model.train_pretext_task(train_loader, gaussian_model.optimizer, gaussian_model.criterion, Config.EPOCHS)
    gaussian_model.save_model(Config.PRETEXT_TASKS['gaussian_blur']['pretext_model_path'])

    # Fine-tuning the Gaussian Blur model for scene classification
    print("Fine-tuning Gaussian Blur model for scene classification...")
    scene_classifier_gaussian = SceneClassifier('EfficientNet-B0', Config.NUM_CLASSES)
    scene_classifier_gaussian.fine_tune_classifier(valid_loader)
    gaussian_finetuned_accuracy = scene_classifier_gaussian.evaluate_model(valid_loader)

    # Experiment tracking
    # Logging the performance and setup details for comparison as per the assignment.
    tracker = ExperimentTracker()
    tracker.log_experiment({
        'model_type': 'Supervised',
        'accuracy': supervised_accuracy,
        'model_details': 'EfficientNet-B0 trained with full supervision.'
    })
    tracker.log_experiment({
        'model_type': 'Self-Supervised with Gaussian Blur Fine-tuned',
        'accuracy': gaussian_finetuned_accuracy,
        'model_details': 'EfficientNet-B0 pre-trained, self-supervised on Gaussian Blur, fine-tuned on scene classification.'
    })

    # Generating the report of experiments
    tracker.report_results()


if __name__ == "__main__":
    main()
