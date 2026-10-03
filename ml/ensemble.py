import tensorflow as tf

from ml.backbones import (
    build_vit_backbone,
    build_efficientnet_backbone,
    build_inception_resnet_backbone,
    extract_vit_features,
    extract_cnn_features,
)
from ml.heads import (
    build_stack_head,
    build_vit_head,
    build_lora_head,
)

NUM_CLASSES = 120


class DogBreedEnsemble(tf.keras.Model):
    """
    TensorFlow/Keras implementation of the three-branch dog-breed ensemble.

    Branch A:
        ViT-L/32 + EfficientNetB3 + InceptionResNetV2
        1024 + 1536 + 1536 = 4096 features
        -> stack classifier

    Branch B:
        ViT-L/32
        1024 features
        -> ViT classifier

    Branch C:
        ViT-L/32
        1024 features
        -> low-rank feature adapter
        -> LoRA-style classifier head

    All backbones are frozen by default.
    """

    def __init__(self):
        super().__init__(name="dog_breed_ensemble")

        # ---------------------------------------------------------
        # Backbones
        # ---------------------------------------------------------
        self.vit_backbone = build_vit_backbone()
        self.efficientnet_backbone = build_efficientnet_backbone()
        self.inception_backbone = build_inception_resnet_backbone()

        # ---------------------------------------------------------
        # Classification heads
        # ---------------------------------------------------------
        self.stack_head = build_stack_head()
        self.vit_head = build_vit_head()
        self.lora_head = build_lora_head()

        # ---------------------------------------------------------
        # Low-rank feature adapter for Branch C
        #
        # This is deliberately implemented as a Keras-native
        # low-rank adapter rather than pretending to have the
        # original notebook's PEFT transformer target modules.
        # ---------------------------------------------------------
        self.lora_down = tf.keras.layers.Dense(
            4,
            use_bias=False,
            name="lora_down",
        )

        self.lora_up = tf.keras.layers.Dense(
            1024,
            use_bias=False,
            kernel_initializer="zeros",
            name="lora_up",
        )

        # Original notebook's final blend configuration.
        self.branch_weights = tf.constant(
            [0.59, 0.13, 0.28],
            dtype=tf.float32,
        )

        self.temperature = tf.Variable(
            0.95,
            trainable=False,
            dtype=tf.float32,
            name="temperature",
        )

    def extract_features(self, images, training=False):
        """
        Run each backbone once and return the three feature branches.
        """

        # ViT
        vit_tokens = self.vit_backbone(images, training=False)
        vit_features = vit_tokens[:, 0, :]

        # CNN branches
        efficientnet_maps = self.efficientnet_backbone(
            images,
            training=False,
        )
        efficientnet_features = tf.reduce_mean(
            efficientnet_maps,
            axis=[1, 2],
        )

        inception_maps = self.inception_backbone(
            images,
            training=False,
        )
        inception_features = tf.reduce_mean(
            inception_maps,
            axis=[1, 2],
        )

        return (
            vit_features,
            efficientnet_features,
            inception_features,
        )

    def call(self, images, training=False):
        """
        Return calibrated ensemble probabilities.
        """

        (
            vit_features,
            efficientnet_features,
            inception_features,
        ) = self.extract_features(images, training=training)

        # ---------------------------------------------------------
        # Branch A: 4096-dimensional stacked representation
        # ---------------------------------------------------------
        stack_features = tf.concat(
            [
                vit_features,
                efficientnet_features,
                inception_features,
            ],
            axis=-1,
        )

        stack_logits = self.stack_head(
            stack_features,
            training=training,
        )

        # ---------------------------------------------------------
        # Branch B: ViT-only classifier
        # ---------------------------------------------------------
        vit_logits = self.vit_head(
            vit_features,
            training=training,
        )

        # ---------------------------------------------------------
        # Branch C: low-rank adapter + classifier
        # ---------------------------------------------------------
        lora_delta = self.lora_up(
            self.lora_down(vit_features),
            training=training,
        )

        adapted_vit_features = vit_features + lora_delta

        lora_logits = self.lora_head(
            adapted_vit_features,
            training=training,
        )

        # ---------------------------------------------------------
        # Weighted ensemble
        # ---------------------------------------------------------
        stack_probs = tf.nn.softmax(stack_logits, axis=-1)
        vit_probs = tf.nn.softmax(vit_logits, axis=-1)
        lora_probs = tf.nn.softmax(lora_logits, axis=-1)

        ensemble_probs = (
            self.branch_weights[0] * stack_probs
            + self.branch_weights[1] * vit_probs
            + self.branch_weights[2] * lora_probs
        )

        # Temperature calibration
        log_probs = tf.math.log(
            tf.clip_by_value(
                ensemble_probs,
                1e-7,
                1.0,
            )
        )

        calibrated_probs = tf.nn.softmax(
            log_probs / self.temperature,
            axis=-1,
        )

        return calibrated_probs

    def branch_outputs(self, images):
        """
        Return individual branch probabilities plus ensemble output.

        Useful during validation and calibration.
        """

        (
            vit_features,
            efficientnet_features,
            inception_features,
        ) = self.extract_features(images)

        stack_features = tf.concat(
            [
                vit_features,
                efficientnet_features,
                inception_features,
            ],
            axis=-1,
        )

        stack_logits = self.stack_head(
            stack_features,
            training=False,
        )

        vit_logits = self.vit_head(
            vit_features,
            training=False,
        )

        lora_delta = self.lora_up(
            self.lora_down(vit_features),
            training=False,
        )

        adapted_vit_features = vit_features + lora_delta

        lora_logits = self.lora_head(
            adapted_vit_features,
            training=False,
        )

        stack_probs = tf.nn.softmax(stack_logits, axis=-1)
        vit_probs = tf.nn.softmax(vit_logits, axis=-1)
        lora_probs = tf.nn.softmax(lora_logits, axis=-1)

        ensemble_probs = (
            self.branch_weights[0] * stack_probs
            + self.branch_weights[1] * vit_probs
            + self.branch_weights[2] * lora_probs
        )

        return {
            "stack": stack_probs,
            "vit": vit_probs,
            "lora": lora_probs,
            "ensemble": ensemble_probs,
        }


def build_ensemble():
    """
    Construct the ensemble model.
    """
    return DogBreedEnsemble()