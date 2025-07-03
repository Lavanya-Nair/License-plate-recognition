import os
import numpy as np
from collections import Counter
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import dump

# All valid license plate characters
letters = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I',
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

def read_training_data(training_directory, target_shape=(20, 20)):
    image_data = []
    target_data = []

    print(f"📁 Loading training data from: {training_directory}")

    for each_letter in letters:
        letter_folder = os.path.join(training_directory, each_letter)
        if not os.path.exists(letter_folder):
            print(f"⚠️ Folder for {each_letter} not found, skipping.")
            continue

        for file_name in os.listdir(letter_folder):
            if not file_name.endswith(".jpg"):
                continue

            image_path = os.path.join(letter_folder, file_name)
            try:
                img = imread(image_path, as_gray=True)
                img_resized = resize(img, target_shape, anti_aliasing=True)

                binary_image = img_resized < threshold_otsu(img_resized)
                flat_image = binary_image.flatten()

                image_data.append(flat_image)
                target_data.append(each_letter)
            except Exception as e:
                print(f"⚠️ Failed to process image {image_path}: {e}")

    print(f"✅ Total samples: {len(image_data)}")
    return np.array(image_data), np.array(target_data)

def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    training_dataset_dir = os.path.join(current_dir, 'train')

    image_data, target_data = read_training_data(training_dataset_dir)

    print(f"\n📊 Class distribution:\n{Counter(target_data)}")

    # Define hyperparameter grid
    param_grid = {
        'svc__kernel': ['linear', 'rbf'],
        'svc__C': [1, 10, 100],
        'svc__gamma': [0.001, 0.01, 0.1, 1]
    }

    # Create a pipeline: StandardScaler + SVM
    pipeline = make_pipeline(StandardScaler(), SVC(probability=True))

    # GridSearchCV for tuning
    grid_search = GridSearchCV(pipeline, param_grid, cv=4, n_jobs=-1, verbose=2)

    print("\n🔍 Running GridSearchCV for hyperparameter tuning...")
    grid_search.fit(image_data, target_data)

    print(f"\n✅ Best Parameters: {grid_search.best_params_}")
    print(f"📈 Best Cross-Validation Accuracy: {grid_search.best_score_ * 100:.2f}%")

    # Save the trained model
    save_path = os.path.join(current_dir, 'models', 'svc')
    os.makedirs(save_path, exist_ok=True)
    model_file = os.path.join(save_path, 'svc_best.pkl')
    dump(grid_search.best_estimator_, model_file)
    print(f"💾 Model saved at: {model_file}")

if __name__ == "__main__":
    main()
