from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.backbones import (
    build_vit_backbone,
    build_efficientnet_backbone,
    build_inception_resnet_backbone,
)
from ml.dataset import (
    load_labels,
    build_filepaths,
    create_stratified_split,
    decode_and_resize,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABELS_PATH = PROJECT_ROOT / "data" / "raw" / "labels.csv"
TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "features"

IMAGE_SIZE = (384, 384)
BATCH_SIZE = 16


def build_image_dataset(filepaths, labels):
    dataset = tf.data.Dataset.from_tensor_slices(
        (filepaths, labels)
    )

    dataset = dataset.map(
        decode_and_resize,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def extract_features(
    dataset,
    vit_backbone,
    efficientnet_backbone,
    inception_backbone,
):
    vit_features = []
    efficientnet_features = []
    inception_features = []
    labels = []

    total_batches = tf.data.experimental.cardinality(dataset).numpy()

    for batch_index, (images, batch_labels) in enumerate(dataset):
        print(
            f"Processing batch "
            f"{batch_index + 1}/{total_batches}"
        )

        vit_tokens = vit_backbone(
            images,
            training=False,
        )

        batch_vit = vit_tokens[:, 0, :]

        efficientnet_maps = efficientnet_backbone(
            images,
            training=False,
        )

        batch_efficientnet = tf.reduce_mean(
            efficientnet_maps,
            axis=[1, 2],
        )

        inception_maps = inception_backbone(
            images,
            training=False,
        )

        batch_inception = tf.reduce_mean(
            inception_maps,
            axis=[1, 2],
        )

        vit_features.append(batch_vit.numpy())
        efficientnet_features.append(
            batch_efficientnet.numpy()
        )
        inception_features.append(
            batch_inception.numpy()
        )
        labels.append(batch_labels.numpy())

    return (
        np.concatenate(vit_features, axis=0),
        np.concatenate(efficientnet_features, axis=0),
        np.concatenate(inception_features, axis=0),
        np.concatenate(labels, axis=0),
    )


def save_features(prefix, features):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        OUTPUT_DIR / f"{prefix}_vit.npy",
        features[0],
    )

    np.save(
        OUTPUT_DIR / f"{prefix}_efficientnet.npy",
        features[1],
    )

    np.save(
        OUTPUT_DIR / f"{prefix}_inception.npy",
        features[2],
    )

    np.save(
        OUTPUT_DIR / f"{prefix}_labels.npy",
        features[3],
    )


def main():
    print("Loading labels...")

    labels_df = load_labels(LABELS_PATH)

    print("Building file paths...")

    filepaths, labels, breed_to_index = build_filepaths(
        labels_df,
        TRAIN_DIR,
    )

    print(f"Total images: {len(filepaths)}")
    print(f"Classes: {len(breed_to_index)}")

    print("Creating stratified split...")

    (
        train_files,
        val_files,
        train_labels,
        val_labels,
    ) = create_stratified_split(
        filepaths,
        labels,
        validation_size=0.10,
    )

    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

    print("\nLoading pretrained backbones...")

    vit_backbone = build_vit_backbone()

    efficientnet_backbone = (
        build_efficientnet_backbone()
    )

    inception_backbone = (
        build_inception_resnet_backbone()
    )

    print("BACKBONES LOADED")

    print("\nExtracting training features...")

    train_dataset = build_image_dataset(
        train_files,
        train_labels,
    )

    train_features = extract_features(
        train_dataset,
        vit_backbone,
        efficientnet_backbone,
        inception_backbone,
    )

    print("\nSaving training features...")

    save_features(
        "train",
        train_features,
    )

    print("\nExtracting validation features...")

    val_dataset = build_image_dataset(
        val_files,
        val_labels,
    )

    val_features = extract_features(
        val_dataset,
        vit_backbone,
        efficientnet_backbone,
        inception_backbone,
    )

    print("\nSaving validation features...")

    save_features(
        "val",
        val_features,
    )

    print("\nFEATURE EXTRACTION COMPLETE")

    print(
        "Train ViT:",
        train_features[0].shape,
    )

    print(
        "Train EfficientNet:",
        train_features[1].shape,
    )

    print(
        "Train InceptionResNet:",
        train_features[2].shape,
    )

    print(
        "Validation ViT:",
        val_features[0].shape,
    )

    print(
        "Validation EfficientNet:",
        val_features[1].shape,
    )

    print(
        "Validation InceptionResNet:",
        val_features[2].shape,
    )


if __name__ == "__main__":
    main()