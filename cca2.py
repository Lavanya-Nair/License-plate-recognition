from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def detect_plate_candidates(binary_image, gray_image=None, debug=False):
    label_image = measure.label(binary_image)
    plate_like_objects = []
    plate_objects_coordinates = []

    # Loosened constraints for plate dimensions
    plate_dimensions = (0.05 * label_image.shape[0], 0.6 * label_image.shape[0],
                        0.10 * label_image.shape[1], 0.9 * label_image.shape[1])
    min_height, max_height, min_width, max_width = plate_dimensions
    min_aspect, max_aspect = 3.5, 6.5  # Narrower range for plates
    min_area = 1000
    max_area = 15000

    candidates = []
    candidate_coords = []
    candidate_aspects = []

    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc
        aspect_ratio = width / float(height)
        area = region.area
        # Only consider regions in the lower half of the image
        if minr < label_image.shape[0] // 2:
            continue
        if (min_height < height < max_height and
            min_width < width < max_width and
            min_aspect < aspect_ratio < max_aspect and
            min_area < area < max_area):
            plate_like_objects.append(binary_image[minr:maxr, minc:maxc])
            plate_objects_coordinates.append((minr, minc, maxr, maxc))
            candidates.append(binary_image[minr:maxr, minc:maxc])
            candidate_coords.append((minr, minc, maxr, maxc))
            candidate_aspects.append(aspect_ratio)
            if debug:
                print(f"Candidate: area={area}, aspect={aspect_ratio:.2f}, bbox=({minr},{minc},{maxr},{maxc})")

    # If multiple candidates, pick the one closest to typical aspect ratio
    if candidates:
        target_ar = 4.5
        best_idx = min(range(len(candidate_aspects)), key=lambda i: abs(candidate_aspects[i] - target_ar))
        best_plate = candidates[best_idx]
        best_coord = candidate_coords[best_idx]
        if debug and gray_image is not None:
            fig, ax = plt.subplots(1)
            ax.imshow(gray_image, cmap='gray')
            minr, minc, maxr, maxc = best_coord
            rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            plt.title('Detected License Plate Region')
            plt.show()
        return [best_plate], [best_coord]
    else:
        if debug:
            print("No plate-like region found.")
        return [], []
