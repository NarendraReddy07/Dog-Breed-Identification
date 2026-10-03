import tensorflow as tf
import keras_hub


IMAGE_SIZE = (384, 384)


def build_vit_backbone():
    """Load pretrained ViT-L/32 backbone."""
    backbone = keras_hub.models.ViTBackbone.from_preset(
        "vit_large_patch32_384_imagenet"
    )
    backbone.trainable = False
    return backbone


def build_efficientnet_backbone():
    """Load pretrained EfficientNetB3 backbone."""
    backbone = tf.keras.applications.EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )
    backbone.trainable = False
    return backbone


def build_inception_resnet_backbone():
    """Load pretrained InceptionResNetV2 backbone."""
    backbone = tf.keras.applications.InceptionResNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )
    backbone.trainable = False
    return backbone


def extract_vit_features(backbone, images):
    """Extract the ViT CLS-token representation."""
    tokens = backbone(images)
    cls_token = tokens[:, 0, :]
    return cls_token


def extract_cnn_features(backbone, images):
    """Apply global average pooling to CNN feature maps."""
    feature_maps = backbone(images, training=False)
    features = tf.reduce_mean(feature_maps, axis=[1, 2])
    return features