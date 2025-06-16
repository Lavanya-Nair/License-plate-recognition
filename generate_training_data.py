import os
import cv2
import xml.etree.ElementTree as ET

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
LABEL_DIR = os.path.join(BASE_DIR, "data", "annotations")
OUTPUT_DIR = os.path.join(BASE_DIR, "train")

def parse_voc_xml(xml_file_path):
    """
    Parses a Pascal VOC-style XML file and returns a list of (label, bbox) tuples.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    characters = []
    for obj in root.findall("object"):
        label = obj.find("name").text.strip()
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        characters.append((label, (xmin, ymin, xmax, ymax)))

    return characters

def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def main():
    print(f"▶️ IMAGE_DIR contains: {os.listdir(IMAGE_DIR)}")
    print(f"▶️ LABEL_DIR contains: {os.listdir(LABEL_DIR)}\n")

    for filename in sorted(os.listdir(IMAGE_DIR)):
        if not filename.endswith(".png"):
            print(f"⏭️ Skipping non-image file: {filename}")
            continue

        img_path = os.path.join(IMAGE_DIR, filename)
        xml_path = os.path.join(LABEL_DIR, filename.replace(".png", ".xml"))

        if not os.path.exists(xml_path):
            print(f"❌ Label file not found: {xml_path}")
            continue

        print(f"\n🔍 Processing image: {filename}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not load image: {img_path}")
            continue

        characters = parse_voc_xml(xml_path)

        for label, (xmin, ymin, xmax, ymax) in characters:
            char_crop = img[ymin:ymax, xmin:xmax]
            if char_crop.size == 0:
                print(f"⚠️ Empty crop for {label} in {filename}, skipping.")
                continue

            label_folder = os.path.join(OUTPUT_DIR, label.upper())
            create_output_dir(label_folder)

            count = len(os.listdir(label_folder))
            save_path = os.path.join(label_folder, f"{label}_{count}.jpg")
            cv2.imwrite(save_path, char_crop)
            print(f"✅ Saved '{label}' to {save_path}")

    print("\n🎉 Done! Check your `train/` folders for character crops.")

if __name__ == "__main__":
    main()
