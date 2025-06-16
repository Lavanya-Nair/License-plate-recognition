import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.transform import resize
import numpy as np
import segmentation  # your segmentation.py module
import joblib
import localization

def preprocess_image(image):
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray_image = rgb2gray(image)
    else:
        gray_image = image
    
    # Normalize the image
    gray_image = (gray_image * 255).astype(np.uint8)
    
    # Apply adaptive thresholding
    thresh = threshold_otsu(gray_image)
    binary = gray_image < thresh
    
    return gray_image, binary

def main():
    # Load model
    model = joblib.load("models/svc/svc_best.pkl")

    # Read image properly before passing
    image_path = "car2.jpg"
    image = imread(image_path)
    
    # Preprocess image
    gray_image, binary_image = preprocess_image(image)
    
    # Display preprocessing results
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.subplot(132)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale')
    plt.subplot(133)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary')
    plt.show()

    # Segment characters
    print("\n🔍 Attempting to segment characters...")
    characters, column_list = segmentation.segment_characters(gray_image, debug=True)

    # Sort characters left to right
    if characters:
        print(f"\n✅ Found {len(characters)} potential characters")
        characters = [char for _, char in sorted(zip(column_list, characters), key=lambda x: x[0])]

        # Display each segmented character
        plt.figure(figsize=(15, 3))
        for i, char_img in enumerate(characters):
            plt.subplot(1, len(characters), i+1)
            plt.imshow(char_img, cmap='gray')
            plt.axis('off')
        plt.suptitle('Segmented Characters')
        plt.show()

        # Predict each character
        plate_number = ""
        for char_img in characters:
            flattened = char_img.reshape(1, -1)
            prediction = model.predict(flattened)[0]
            plate_number += prediction
            print(f"Character predicted: {prediction}")

        print(f"\n✅ Final License Plate Prediction: {plate_number}")
    else:
        print("\n❌ No characters detected. Try adjusting the segmentation parameters.")

if __name__ == "__main__":
    main()
