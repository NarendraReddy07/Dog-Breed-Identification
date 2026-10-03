import tensorflow as tf

from ml.ensemble import build_ensemble


def main():
    print("Building ensemble...")

    model = build_ensemble()

    dummy_images = tf.random.uniform(
        shape=(1, 384, 384, 3),
        dtype=tf.float32,
    )

    print("Running forward pass...")

    predictions = model(dummy_images, training=False)

    print("Prediction shape:", predictions.shape)

    assert predictions.shape == (1, 120), (
        f"Unexpected prediction shape: {predictions.shape}"
    )

    probability_sum = tf.reduce_sum(predictions, axis=-1)

    print(
        "Probability sum:",
        probability_sum.numpy(),
    )

    assert tf.reduce_all(
        tf.abs(probability_sum - 1.0) < 1e-5
    )

    print("ENSEMBLE ARCHITECTURE OK")


if __name__ == "__main__":
    main()