from torchvision.transforms import functional as F
import random


def gaussian_blur(image, label):
    """
    Apply Gaussian blur to an image with a randomly chosen kernel size.
    """
    kernel_size = [5, 9, 13, 17, 21][label]
    return F.gaussian_blur(image, kernel_size=[kernel_size, kernel_size])


def perturbation(image, label, size=10):
    """
    Apply a random black or white square perturbation to the image.
    """
    start_x, start_y = random.randint(0, image.shape[1] - size), random.randint(0, image.shape[2] - size)
    end_x, end_y = start_x + size, start_y + size
    perturbed_img = image.clone()
    perturbed_img[:, start_x:end_x, start_y:end_y] = 0 if label == 0 else 1
    return perturbed_img
