from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import keras_hub


BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"

IMAGE_SIZE = (384, 384)
NUM_CLASSES = 120


class ModelManager:
    """
    Loads the production dog-breed models once and performs
    deterministic inference.
    """

    def __init__(self):
        self.loaded = False

        self.class_mapping = None
        self.temperature = 1.10

        self.vit_backbone = None
        self.efficientnet_backbone = None
        self.inception_backbone = None

        self.stack_heads = []

    def load(self):
        if self.loaded:
            return

        print("Loading dog-breed production models...")

        self._load_config()
        self._load_class_mapping()
        self._load_backbones()
        self._load_stack_heads()

        self.loaded = True

        print("Dog-breed production models loaded successfully.")

    def _load_config(self):
        config_path = ARTIFACTS_DIR / "production_config.json"

        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        self.temperature = float(config["temperature"])

        print(f"Model version: {config['model_version']}")
        print(f"Production branch: {config['production_branch']}")
        print(f"Temperature: {self.temperature}")

    def _load_class_mapping(self):
        mapping_path = ARTIFACTS_DIR / "class_mapping.json"

        with open(mapping_path, "r", encoding="utf-8") as file:
            mapping = json.load(file)

        if "index_to_breed" in mapping:
            index_to_breed = mapping["index_to_breed"]
        else:
            index_to_breed = mapping

        self.class_mapping = {
            int(index): breed
            for index, breed in index_to_breed.items()
        }

        if len(self.class_mapping) != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} classes, "
                f"found {len(self.class_mapping)}"
            )

        print(f"Loaded {len(self.class_mapping)} breed classes.")

    def _load_backbones(self):
        print("Loading ViT-L/32...")

        self.vit_backbone = keras_hub.models.ViTBackbone.from_preset(
            "vit_large_patch32_384_imagenet"
        )

        self.vit_backbone.trainable = False

        print("Loading EfficientNetB3...")

        self.efficientnet_backbone = (
            tf.keras.applications.EfficientNetB3(
                weights="imagenet",
                include_top=False,
                input_shape=(384, 384, 3),
            )
        )

        self.efficientnet_backbone.trainable = False

        print("Loading InceptionResNetV2...")

        self.inception_backbone = (
            tf.keras.applications.InceptionResNetV2(
                weights="imagenet",
                include_top=False,
                input_shape=(384, 384, 3),
            )
        )

        self.inception_backbone.trainable = False

    def _load_stack_heads(self):
        print("Loading stack classifier heads...")

        for seed in [1234, 2345, 3456, 4567, 5678]:
            model_path = (
                ARTIFACTS_DIR
                / f"stack_seed_{seed}.keras"
            )

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Missing model artifact: {model_path}"
                )

            head = tf.keras.models.load_model(
                model_path,
                compile=False,
            )

            self.stack_heads.append(head)

        print(
            f"Loaded {len(self.stack_heads)} stack heads."
        )

    def _preprocess_vit(self, image):
        image = tf.cast(image, tf.float32)
        image = image / 255.0

        mean = tf.constant(
            [0.485, 0.456, 0.406],
            dtype=tf.float32,
        )

        std = tf.constant(
            [0.229, 0.224, 0.225],
            dtype=tf.float32,
        )

        return (image - mean) / std

    def _preprocess_inception(self, image):
        return (
            tf.keras.applications.inception_resnet_v2
            .preprocess_input(image)
        )

    def _extract_features(self, image):
        """
        Extract the three feature vectors required by
        the production stack branch.
        """

        # ViT
        vit_image = self._preprocess_vit(image)

        vit_tokens = self.vit_backbone(
            vit_image,
            training=False,
        )

        vit_features = vit_tokens[:, 0, :]

        # EfficientNetB3
        efficientnet_features = (
            self.efficientnet_backbone(
                image,
                training=False,
            )
        )

        efficientnet_features = tf.reduce_mean(
            efficientnet_features,
            axis=[1, 2],
        )

        # InceptionResNetV2
        inception_image = self._preprocess_inception(
            image
        )

        inception_features = (
            self.inception_backbone(
                inception_image,
                training=False,
            )
        )

        inception_features = tf.reduce_mean(
            inception_features,
            axis=[1, 2],
        )

        # 1024 + 1536 + 1536 = 4096
        return tf.concat(
            [
                vit_features,
                efficientnet_features,
                inception_features,
            ],
            axis=-1,
        )

    def predict(self, image):
        """
        Run deterministic production inference.

        Input:
            image: Tensor with shape (1, 384, 384, 3)

        Returns:
            Top-5 breed predictions.
        """

        if not self.loaded:
            self.load()

        features = self._extract_features(image)

        head_probabilities = []

        for head in self.stack_heads:
            logits = head(
                features,
                training=False,
            )

            probabilities = tf.nn.softmax(
                logits,
                axis=-1,
            )

            head_probabilities.append(
                probabilities
            )

        # Average the five independently trained heads.
        ensemble_probabilities = tf.reduce_mean(
            tf.stack(
                head_probabilities,
                axis=0,
            ),
            axis=0,
        )

        # Production temperature calibration.
        log_probabilities = tf.math.log(
            tf.clip_by_value(
                ensemble_probabilities,
                1e-7,
                1.0,
            )
        )

        calibrated_probabilities = tf.nn.softmax(
            log_probabilities / self.temperature,
            axis=-1,
        )

        probabilities = (
            calibrated_probabilities[0].numpy()
        )

        top_indices = np.argsort(
            probabilities
        )[::-1][:5]

        predictions = []

        for index in top_indices:
            predictions.append(
                {
                    "breed": self.class_mapping[int(index)],
                    "confidence": float(
                        probabilities[index]
                    ),
                }
            )

        return predictions


model_manager = ModelManager()