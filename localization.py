from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

def convert_to_grayscale(image_path):
    car_image = imread(image_path, as_gray=True)
    gray_car_image = car_image * 255
    return gray_car_image

def binarize_image(gray_image):
    threshold_value = threshold_otsu(gray_image)
    binary_image = gray_image > threshold_value
    return binary_image
