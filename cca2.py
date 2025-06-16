from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def detect_plate_candidates(binary_image, gray_image=None, debug=False):
    label_image = measure.label(binary_image)
    plate_like_objects = []
    plate_objects_coordinates = []

    plate_dimensions = (0.08 * label_image.shape[0], 0.2 * label_image.shape[0],
                        0.15 * label_image.shape[1], 0.4 * label_image.shape[1])
    min_height, max_height, min_width, max_width = plate_dimensions

    if debug and gray_image is not None:
        fig, ax1 = plt.subplots(1)
        ax1.imshow(gray_image, cmap="gray")

    for region in regionprops(label_image):
        if region.area < 50:
            continue

        minr, minc, maxr, maxc = region.bbox
        region_height = maxr - minr
        region_width = maxc - minc

        if (min_height <= region_height <= max_height and
            min_width <= region_width <= max_width and
            region_width > region_height):
            plate_like_objects.append(binary_image[minr:maxr, minc:maxc])
            plate_objects_coordinates.append((minr, minc, maxr, maxc))
            if debug and gray_image is not None:
                rectBorder = patches.Rectangle((minc, minr), region_width, region_height,
                                               edgecolor="red", linewidth=2, fill=False)
                ax1.add_patch(rectBorder)

    if debug and gray_image is not None:
        plt.title("Detected License Plate Region")
        plt.show()

    return plate_like_objects, plate_objects_coordinates
    plt.imshow(plate_like_objects[0], cmap='gray')
    plt.title("Detected Plate Region")
    plt.show()
