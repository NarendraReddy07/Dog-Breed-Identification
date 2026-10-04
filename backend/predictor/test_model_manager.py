from pathlib import Path
import sys

import tensorflow as tf

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)

from backend.predictor.model_manager import ModelManager


print("Creating ModelManager...")

manager = ModelManager()

print("Loading production model...")

manager.load()

print("Model loaded:", manager.loaded)
print("Number of stack heads:", len(manager.stack_heads))

dummy_image = tf.zeros(
    (1, 384, 384, 3),
    dtype=tf.float32,
)

print("Running prediction...")

predictions = manager.predict(dummy_image)

print("\nTop 5 predictions:")

for position, prediction in enumerate(
    predictions,
    start=1,
):
    print(
        f"{position}. "
        f"{prediction['breed']}: "
        f"{prediction['confidence'] * 100:.2f}%"
    )

print("\nMODEL MANAGER TEST PASSED")