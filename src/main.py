import torch

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
    gaussian_config = Config.PRETEXT_TASKS['gaussian_blur']
    gaussian_task = PretextTask('gaussian_blur')
    # gaussian_task.train_pretext_task(train_loader, gaussian_task.optimizer, gaussian_task.criterion, Config.EPOCHS)
    gaussian_task.train_model(train_loader, valid_loader, Config.EPOCHS)
    gaussian_task.save_model(Config.PRETEXT_TASKS['gaussian_blur']['pretext_model_path'])

    image = data_handler.load_image('15-Scene/00/1.jpg')
    transformed_image, predicted_label, true_label = gaussian_task.classify_image(image)
    print(f"Predicted label: {predicted_label}, True label: {true_label}")
    data_handler.save_transformed_image(transformed_image, 'gaussian_blur_transformed.jpg')

    gaussian_task.prepare_for_scene_classification()
    gaussian_task.fine_tune_classifier(train_loader, valid_loader, Config.EPOCHS)
    gaussian_finetuned_accuracy = gaussian_task.evaluate_model(valid_loader)
    gaussian_task.save_model(Config.PRETEXT_TASKS['gaussian_blur']['scene_classifier_model_path'])

    # Black and White Perturbation Pretext Task
    print("Training on Black and White Perturbation pretext task...")
    bw_perturbation_task = PretextTask('black_white_perturbation')
    # bw_perturbation_task.train_pretext_task(train_loader, bw_perturbation_task.optimizer, bw_perturbation_task.criterion, Config.EPOCHS)
    bw_perturbation_task.train_model(train_loader, valid_loader, Config.EPOCHS)
    bw_perturbation_task.save_model(Config.PRETEXT_TASKS['black_white_perturbation']['pretext_model_path'])

    image = data_handler.load_image('15-Scene/00/1.jpg')
    transformed_image, predicted_label, true_label = bw_perturbation_task.classify_image(image)
    print(f"Predicted label: {predicted_label}, True label: {true_label}")
    data_handler.save_transformed_image(transformed_image, 'pertubation_transformed.jpg')

    bw_perturbation_task.prepare_for_scene_classification()
    bw_perturbation_task.fine_tune_classifier(train_loader, valid_loader, Config.EPOCHS)
    bw_finetuned_accuracy = bw_perturbation_task.evaluate_model(valid_loader)
    bw_perturbation_task.save_model(Config.PRETEXT_TASKS['black_white_perturbation']['scene_classifier_model_path'])

    # Experiment tracking
    tracker = ExperimentTracker()
    # tracker.log_experiment({
    #     'model_type': 'Supervised',
    #     'accuracy': supervised_accuracy,
    #     'model_details': 'EfficientNet-B0 trained with full supervision.'
    # })
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
