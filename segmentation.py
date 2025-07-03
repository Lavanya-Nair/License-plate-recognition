import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.transform import resize
import numpy as np
import matplotlib.patches as patches

def segment_characters(plate_img, debug=True):
    # Threshold image
    binary_plate = plate_img < 128  # works if image is grayscale

    if debug:
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(plate_img, cmap='gray')
        plt.title('Original Grayscale')
        plt.subplot(132)
        plt.imshow(binary_plate, cmap='gray')
        plt.title('Binary Image')

    # Label connected components
    label_img = label(binary_plate)
    regions = regionprops(label_img)

    character_images = []
    column_list = []
    
    # Create a copy of binary image for visualization
    debug_img = np.copy(binary_plate)

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        roi = binary_plate[minr:maxr, minc:maxc]
        area = region.area
        height, width = roi.shape
        aspect_ratio = width / float(height)

        # Adjusted parameters for better detection
        if (50 < area < 1500 and  # Increased area range
            0.1 < aspect_ratio < 1.5 and  # More flexible aspect ratio
            height > 8 and width > 3):  # Reduced minimum dimensions
            
            resized_char = resize(roi.astype(float), (20, 20), anti_aliasing=True)
            character_images.append(resized_char)
            column_list.append(minc)
            
            # Draw rectangle around detected character for debugging
            if debug:
                debug_img[minr:maxr, minc:maxc] = 2  # Mark detected regions
                print(f"🔍 Character found at column {minc}, area: {area:.1f}, AR: {aspect_ratio:.2f}")

    if debug:
        plt.subplot(133)
        plt.imshow(debug_img, cmap='gray')
        plt.title('Detected Characters')
        # Draw rectangles on the original grayscale image for each detected character
        ax = plt.gca()
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            area = region.area
            height = maxr - minr
            width = maxc - minc
            aspect_ratio = width / float(height)
            if (50 < area < 1500 and 0.1 < aspect_ratio < 1.5 and height > 8 and width > 3):
                rect = patches.Rectangle((minc, minr), width, height, linewidth=1.5, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
        plt.tight_layout()
        plt.show()

    return character_images, column_list
