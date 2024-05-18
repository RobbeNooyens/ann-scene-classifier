import torch

from config import Config
from data_handler import DataHandler
from model_manager import ModelManager
from pretext_task import PretextTask
from scene_classifier import SceneClassifier


def main():
    # Data preparation
    data_handler = DataHandler()
    train_loader, valid_loader = data_handler.load_data()

    # # Supervised Learning
    # print("Training supervised model...")
    # supervised_model = ModelManager('EfficientNet-B0', checkpoint=Config.MODEL_SAVE_PATH)
    # supervised_model.train_model(train_loader, valid_loader, epochs=Config.EPOCHS)
    # supervised_accuracy, _ = supervised_model.evaluate_model(valid_loader)
    #
    # Gaussian Blur Pretext Task
    print("Training on Gaussian Blur pretext task...")
    gaussian_task = PretextTask('gaussian_blur')
    gaussian_task.train_model(train_loader, valid_loader, Config.EPOCHS)
    #
    # # image = data_handler.load_image('15-Scene/00/1.jpg')
    # # transformed_image, predicted_label, true_label = gaussian_task.classify_image(image)
    # # print(f"Predicted label: {predicted_label}, True label: {true_label}")
    # # data_handler.save_transformed_image(transformed_image, 'gaussian_blur_transformed.jpg')
    #
    # gaussian_task.prepare_for_scene_classification()
    # gaussian_task.train_model(train_loader, valid_loader, Config.EPOCHS)
    # gaussian_finetuned_accuracy = gaussian_task.evaluate_model(valid_loader)

    # Black and White Perturbation Pretext Task
    print("Training on Black and White Perturbation pretext task...")
    bw_perturbation_task = PretextTask('black_white_perturbation')
    bw_perturbation_task.train_model(train_loader, valid_loader, 10)

    # image = data_handler.load_image('15-Scene/00/1.jpg')
    # transformed_image, predicted_label, true_label = bw_perturbation_task.classify_image(image)
    # print(f"Predicted label: {predicted_label}, True label: {true_label}")
    # data_handler.save_transformed_image(transformed_image, 'pertubation_transformed.jpg')

    bw_perturbation_task.prepare_for_scene_classification()
    bw_perturbation_task.train_model(train_loader, valid_loader, Config.EPOCHS)
    bw_finetuned_accuracy = bw_perturbation_task.evaluate_model(valid_loader)


if __name__ == "__main__":
    main()
