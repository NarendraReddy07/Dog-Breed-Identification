import tensorflow as tf


NUM_CLASSES = 120


def build_classifier_head(
    input_dim: int,
    dropout_rate: float,
    name: str,
) -> tf.keras.Model:
    """Build a classification head for one ensemble branch."""
    inputs = tf.keras.Input(shape=(input_dim,), name=f"{name}_features")

    x = tf.keras.layers.Dropout(
        dropout_rate,
        name=f"{name}_dropout",
    )(inputs)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        name=f"{name}_classifier",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=name,
    )


def build_stack_head() -> tf.keras.Model:
    """4096 → 120 classification head."""
    return build_classifier_head(
        input_dim=4096,
        dropout_rate=0.8270,
        name="stack_head",
    )


def build_vit_head() -> tf.keras.Model:
    """1024 → 120 ViT classification head."""
    return build_classifier_head(
        input_dim=1024,
        dropout_rate=0.6882,
        name="vit_head",
    )


def build_lora_head() -> tf.keras.Model:
    """1024 → 120 LoRA classification head."""
    return build_classifier_head(
        input_dim=1024,
        dropout_rate=0.8502,
        name="lora_head",
    )