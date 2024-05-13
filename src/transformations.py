from torchvision.transforms import functional as F
import random

class ImageTransformer:
    def __init__(self):
        # Define any constants or initialization parameters here if needed
        pass

    def gaussian_blur(self, image, kernel_size=3):
        """
        Apply Gaussian blur to an image with a randomly chosen kernel size.
        """
        return F.gaussian_blur(image, kernel_size=[kernel_size, kernel_size])

    def perturbation(self, image, color=0, size=10):
        """
        Apply a random black or white square perturbation to the image.
        """
        start_x, start_y = random.randint(0, image.shape[1] - size), random.randint(0, image.shape[2] - size)
        end_x, end_y = start_x + size, start_y + size
        perturbed_img = image.clone()
        perturbed_img[:, start_x:end_x, start_y:end_y] = 0 if color == 0 else 1
        return perturbed_img

