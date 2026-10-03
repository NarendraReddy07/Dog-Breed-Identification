from pathlib import Path

import pandas as pd
import tensorflow as tf


IMAGE_SIZE = (384, 384)
NUM_CLASSES = 120


def load_labels(labels_path: str | Path) -> pd.DataFrame:
    """Load the image ID to breed mapping."""
    df = pd.read_csv(labels_path)

    required_columns = {"id", "breed"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["id"].duplicated().any():
        raise ValueError("Duplicate image IDs found in labels.csv")

    if df["breed"].nunique() != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} breeds, found {df['breed'].nunique()}"
        )

    return df


def build_class_mapping(labels_df: pd.DataFrame):
    """Create stable breed-to-index and index-to-breed mappings."""
    breeds = sorted(labels_df["breed"].unique())

    breed_to_index = {
        breed: index for index, breed in enumerate(breeds)
    }

    index_to_breed = {
        index: breed for breed, index in breed_to_index.items()
    }

    return breed_to_index, index_to_breed


def build_filepaths(
    labels_df: pd.DataFrame,
    train_dir: str | Path,
):
    """Create image paths and integer labels."""
    train_dir = Path(train_dir)

    filepaths = []
    labels = []

    breed_to_index, _ = build_class_mapping(labels_df)

    for _, row in labels_df.iterrows():
        image_path = train_dir / f"{row['id']}.jpg"

        if not image_path.exists():
            raise FileNotFoundError(
                f"Training image not found: {image_path}"
            )

        filepaths.append(str(image_path))
        labels.append(breed_to_index[row["breed"]])

    return filepaths, labels, breed_to_index


def decode_and_resize(image_path, label):
    """Read, decode and resize one image."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    return image, label


def build_dataset(
    filepaths,
    labels,
    batch_size=32,
    shuffle=False,
):
    """Build a TensorFlow dataset."""
    dataset = tf.data.Dataset.from_tensor_slices(
        (filepaths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(filepaths),
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        decode_and_resize,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset



from sklearn.model_selection import train_test_split


def create_stratified_split(filepaths, labels, validation_size=0.10):
    """Create a stratified train/validation split."""
    train_files, val_files, train_labels, val_labels = train_test_split(
        filepaths,
        labels,
        test_size=validation_size,
        random_state=42,
        stratify=labels,
    )

    return (
        train_files,
        val_files,
        train_labels,
        val_labels,
    )