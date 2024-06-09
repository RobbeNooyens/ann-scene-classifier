import torch

from config import FINETUNE_CONFIG, GAUSSIAN_BLUR_CONFIG, PERTURBATION_CONFIG, GAUSSIAN_BLUR_SCENES_CONFIG, \
    PERTURBATION_SCENES_CONFIG
from data import DataHandler
from models import ModelManager, PretextTask


def main():
    # Data handlers
    data_handler = DataHandler(FINETUNE_CONFIG)
    gaussian_data_handler = DataHandler(GAUSSIAN_BLUR_CONFIG)
    perturbation_data_handler = DataHandler(PERTURBATION_CONFIG)

    # Load data
    train_loader, valid_loader, test_loader = data_handler.load()
    gaussian_train_loader, gaussian_valid_loader, gaussian_test_loader = gaussian_data_handler.load()
    perturbation_train_loader, perturbation_valid_loader, perturbation_test_loader = perturbation_data_handler.load()

    # Supervised Learning
    print("Training supervised model...")
    supervised_model = ModelManager(FINETUNE_CONFIG)
    supervised_model.train_model(train_loader, valid_loader)
    supervised_model.evaluate_test_set(test_loader)

    # Gaussian Blur Pretext Task
    print("Training on Gaussian Blur pretext task...")
    gaussian_task = PretextTask(GAUSSIAN_BLUR_CONFIG, GAUSSIAN_BLUR_SCENES_CONFIG)
    gaussian_task.train_pretext_model(gaussian_train_loader, gaussian_valid_loader)
    gaussian_task.evaluate_test_set(gaussian_test_loader)
    gaussian_task.train_scene_model(train_loader, valid_loader)
    gaussian_task.evaluate_test_set(test_loader)

    # Black and White Perturbation Pretext Task
    print("Training on Black and White Perturbation pretext task...")
    perturbation_task = PretextTask(PERTURBATION_CONFIG, PERTURBATION_SCENES_CONFIG)
    perturbation_task.train_model(perturbation_train_loader, perturbation_valid_loader)
    perturbation_task.evaluate_test_set(perturbation_test_loader)
    perturbation_task.train_scene_model(train_loader, valid_loader)
    perturbation_task.evaluate_test_set(test_loader)


if __name__ == "__main__":
    main()
