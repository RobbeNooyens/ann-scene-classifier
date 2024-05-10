from config import Config
from data_handler import DataHandler
from model_manager import ModelManager
from pretext_task import PretextTask
from scene_classifier import SceneClassifier
from experiment_tracker import ExperimentTracker


def main():
    # Data preparation
    data_handler = DataHandler()
    train_loader, valid_loader = data_handler.load_data()

    # Supervised Learning
    # print("Training supervised model...")
    # supervised_model = ModelManager('EfficientNet-B0', pretrained=True, use_checkpoint=True)
    # supervised_model.train_model(train_loader, valid_loader, epochs=Config.EPOCHS)
    # supervised_accuracy, _ = supervised_model.evaluate_model(valid_loader)
    # supervised_model.save_model(Config.MODEL_SAVE_PATH)

    # Gaussian Blur Pretext Task
    print("Training on Gaussian Blur pretext task...")
    gaussian_task = PretextTask('EfficientNet-B0', 'gaussian_blur', Config.PRETEXT_TASKS['gaussian_blur']['classes'])
    gaussian_task.train_pretext_task(train_loader, gaussian_task.optimizer, gaussian_task.criterion, Config.EPOCHS)
    gaussian_task.save_model(Config.PRETEXT_TASKS['gaussian_blur']['pretext_model_path'])

    # Black and White Perturbation Pretext Task
    print("Training on Black and White Perturbation pretext task...")
    bw_perturbation_task = PretextTask('EfficientNet-B0', 'black_white_perturbation', Config.PRETEXT_TASKS['black_white_perturbation']['classes'])
    bw_perturbation_task.train_pretext_task(train_loader, bw_perturbation_task.optimizer, bw_perturbation_task.criterion, Config.EPOCHS)
    bw_perturbation_task.save_model(Config.PRETEXT_TASKS['black_white_perturbation']['pretext_model_path'])

    # Fine-tuning Gaussian Blur Model for scene classification
    print("Fine-tuning Gaussian Blur model for scene classification...")
    scene_classifier_gaussian = SceneClassifier('EfficientNet-B0', Config.NUM_CLASSES)
    scene_classifier_gaussian.fine_tune_classifier(valid_loader)
    gaussian_finetuned_accuracy = scene_classifier_gaussian.evaluate_model(valid_loader)

    # Fine-tuning Black and White Perturbation Model for scene classification
    print("Fine-tuning Black and White Perturbation model for scene classification...")
    scene_classifier_bw = SceneClassifier('EfficientNet-B0', Config.NUM_CLASSES)
    scene_classifier_bw.fine_tune_classifier(valid_loader)
    bw_finetuned_accuracy = scene_classifier_bw.evaluate_model(valid_loader)

    # Experiment tracking
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
    tracker.log_experiment({
        'model_type': 'Self-Supervised with Black and White Perturbation Fine-tuned',
        'accuracy': bw_finetuned_accuracy,
        'model_details': 'EfficientNet-B0 pre-trained, self-supervised on Black and White Perturbation, fine-tuned on scene classification.'
    })

    # Generating the report of experiments
    tracker.report_results()


if __name__ == "__main__":
    main()
