import cv2
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from sklearn.decomposition import PCA

# =========================
# CONFIG
# =========================
DATASET_DIR = Path("../garbage-dataset/realwaste-main/RealWaste")
OUTPUT_DIR = Path("../output_v2")

IMAGE_SIZE = (224, 224)

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

LBP_RADIUS = 2
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

TEST_SIZE = 0.2
RANDOM_STATE = 42
PCA_COMPONENTS = 300


# =========================
# HELPER FUNCTIONS
# =========================
def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_image_file(file_path: Path) -> bool:
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return file_path.suffix.lower() in valid_extensions


def load_and_resize_image(image_path: Path, image_size=(224, 224)):
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size)
    return img


def extract_hsv_histogram(image, bins=(16, 16, 16)):
    """
    Extract normalized HSV histogram feature.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_hog_features(image):
    """
    Extract HOG features from grayscale image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )
    return features


def extract_lbp_features(image):
    """
    Extract normalized LBP histogram.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lbp = local_binary_pattern(
        gray,
        P=LBP_N_POINTS,
        R=LBP_RADIUS,
        method=LBP_METHOD
    )

    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)
    return hist


def extract_edge_features(image):
    """
    Extract simple edge density feature from Canny edges.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.array([edges.mean() / 255.0], dtype=np.float32)
    return edge_density


def extract_combined_features(image):
    """
    Combine HSV histogram + HOG + LBP + edge density.
    """
    hsv_feat = extract_hsv_histogram(image)
    hog_feat = extract_hog_features(image)
    lbp_feat = extract_lbp_features(image)
    edge_feat = extract_edge_features(image)

    combined = np.hstack([hsv_feat, hog_feat, lbp_feat, edge_feat])
    return combined


# =========================
# DATASET PROCESSING
# =========================
def process_dataset(dataset_dir: Path):
    features = []
    labels = []
    image_paths = []
    class_counts = {}

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_folders = sorted([f for f in dataset_dir.iterdir() if f.is_dir()])

    print("\nFound class folders:")
    for folder in class_folders:
        print("-", folder.name)

    for class_folder in class_folders:
        class_name = class_folder.name
        class_counts[class_name] = 0

        for file_path in class_folder.iterdir():
            if not is_image_file(file_path):
                continue

            image = load_and_resize_image(file_path, IMAGE_SIZE)
            if image is None:
                print(f"[WARNING] Could not read image: {file_path}")
                continue

            try:
                feat = extract_combined_features(image)
                features.append(feat)
                labels.append(class_name)
                image_paths.append(str(file_path))
                class_counts[class_name] += 1
            except Exception as e:
                print(f"[ERROR] Feature extraction failed for {file_path}: {e}")

    print("\nClass-wise image counts:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)

    return X, y, image_paths, class_counts


def save_class_distribution(class_counts):
    df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Image_Count"])
    df.to_csv(OUTPUT_DIR / "class_distribution.csv", index=False)
    print("\nSaved class distribution to output_v2/class_distribution.csv")


def encode_labels(y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    label_mapping = pd.DataFrame({
        "Class_Name": encoder.classes_,
        "Encoded_Label": range(len(encoder.classes_))
    })
    label_mapping.to_csv(OUTPUT_DIR / "label_mapping.csv", index=False)

    joblib.dump(encoder, OUTPUT_DIR / "label_encoder.pkl")

    print("Saved label mapping and label encoder.")
    return y_encoded, encoder


def split_scale_normalize_pca(X, y_encoded):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    # 1. Standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
    print("Saved scaler.")

    # 2. L2 normalization
    normalizer = Normalizer(norm="l2")
    X_train_norm = normalizer.fit_transform(X_train_scaled)
    X_test_norm = normalizer.transform(X_test_scaled)
    joblib.dump(normalizer, OUTPUT_DIR / "normalizer.pkl")
    print("Saved normalizer.")

    # 3. PCA
    n_components = min(PCA_COMPONENTS, X_train_norm.shape[0], X_train_norm.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_test_pca = pca.transform(X_test_norm)
    joblib.dump(pca, OUTPUT_DIR / "pca.pkl")
    print(f"Saved PCA with {n_components} components.")

    return (
        X_train_scaled, X_test_scaled,
        X_train_norm, X_test_norm,
        X_train_pca, X_test_pca,
        y_train, y_test
    )


def save_processed_data(
    X, y, image_paths,
    X_train_scaled, X_test_scaled,
    X_train_norm, X_test_norm,
    X_train_pca, X_test_pca,
    y_train, y_test
):
    np.save(OUTPUT_DIR / "X_features.npy", X)
    np.save(OUTPUT_DIR / "y_labels.npy", y)

    np.save(OUTPUT_DIR / "X_train_scaled.npy", X_train_scaled)
    np.save(OUTPUT_DIR / "X_test_scaled.npy", X_test_scaled)

    np.save(OUTPUT_DIR / "X_train_norm.npy", X_train_norm)
    np.save(OUTPUT_DIR / "X_test_norm.npy", X_test_norm)

    np.save(OUTPUT_DIR / "X_train_pca.npy", X_train_pca)
    np.save(OUTPUT_DIR / "X_test_pca.npy", X_test_pca)

    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    pd.DataFrame({"image_path": image_paths}).to_csv(
        OUTPUT_DIR / "image_paths.csv", index=False
    )

    print("Saved processed numpy arrays and image paths.")


def main():
    print("Starting improved preprocessing...")
    ensure_output_dir()

    X, y, image_paths, class_counts = process_dataset(DATASET_DIR)

    if len(X) == 0:
        raise ValueError("No valid images found in dataset folder.")

    print(f"\nTotal samples processed: {len(X)}")
    print(f"Original feature vector shape: {X.shape}")

    save_class_distribution(class_counts)

    y_encoded, encoder = encode_labels(y)

    (
        X_train_scaled, X_test_scaled,
        X_train_norm, X_test_norm,
        X_train_pca, X_test_pca,
        y_train, y_test
    ) = split_scale_normalize_pca(X, y_encoded)

    save_processed_data(
        X, y_encoded, image_paths,
        X_train_scaled, X_test_scaled,
        X_train_norm, X_test_norm,
        X_train_pca, X_test_pca,
        y_train, y_test
    )

    print("\nImproved preprocessing completed successfully.")
    print("Output files saved inside the 'output_v2' folder.")


if __name__ == "__main__":
    main()