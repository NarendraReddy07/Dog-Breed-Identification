from pathlib import Path

from dataset import (
    NUM_CLASSES,
    build_dataset,
    build_filepaths,
    create_stratified_split,
    load_labels,
)


ROOT = Path(__file__).resolve().parent.parent

LABELS_PATH = ROOT / "data" / "raw" / "labels.csv"
TRAIN_DIR = ROOT / "data" / "raw" / "train"


labels_df = load_labels(LABELS_PATH)

filepaths, labels, breed_to_index = build_filepaths(
    labels_df,
    TRAIN_DIR,
)

train_files, val_files, train_labels, val_labels = create_stratified_split(
    filepaths,
    labels,
)


print("Total images:", len(filepaths))
print("Training images:", len(train_files))
print("Validation images:", len(val_files))
print("Classes:", len(breed_to_index))

print("")

print("Training dataset:")
train_dataset = build_dataset(
    train_files,
    train_labels,
    batch_size=4,
    shuffle=True,
)

train_images, train_batch_labels = next(iter(train_dataset))

print("Image shape:", train_images.shape)
print("Label shape:", train_batch_labels.shape)

print("")

print("Validation dataset:")
val_dataset = build_dataset(
    val_files,
    val_labels,
    batch_size=4,
    shuffle=False,
)

val_images, val_batch_labels = next(iter(val_dataset))

print("Image shape:", val_images.shape)
print("Label shape:", val_batch_labels.shape)

assert len(filepaths) == 10222
assert len(train_files) + len(val_files) == 10222
assert len(train_files) == 9199
assert len(val_files) == 1023
assert len(breed_to_index) == NUM_CLASSES

assert train_images.shape == (4, 384, 384, 3)
assert val_images.shape == (4, 384, 384, 3)

print("")
print("STRATIFIED DATA SPLIT OK")