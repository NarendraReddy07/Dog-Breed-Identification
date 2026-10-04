import time
from io import BytesIO

import tensorflow as tf
from PIL import Image
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .model_manager import model_manager


class PredictView(APIView):
    """
    Predict the breed of an uploaded dog image.
    """

    def post(self, request):
        uploaded_file = request.FILES.get("image")

        if uploaded_file is None:
            return Response(
                {
                    "error": "No image provided. Use the 'image' field."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        start_time = time.perf_counter()

        try:
            image_bytes = uploaded_file.read()

            image = Image.open(
                BytesIO(image_bytes)
            ).convert("RGB")

            original_width, original_height = image.size

            image = image.resize(
                (384, 384)
            )

            image_array = tf.keras.utils.img_to_array(
                image
            )

            image_tensor = tf.convert_to_tensor(
                image_array,
                dtype=tf.float32,
            )

            image_tensor = tf.expand_dims(
                image_tensor,
                axis=0,
            )

            predictions = model_manager.predict(
                image_tensor
            )

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            return Response(
                {
                    "model_version": "tf-ensemble-v1",
                    "image": {
                        "filename": uploaded_file.name,
                        "width": original_width,
                        "height": original_height,
                        "processed_size": [384, 384],
                    },
                    "predictions": predictions,
                    "top_prediction": predictions[0],
                    "deterministic_inference": True,
                    "tta": False,
                    "mc_dropout": False,
                    "latency_ms": round(
                        latency_ms,
                        2,
                    ),
                },
                status=status.HTTP_200_OK,
            )

        except Exception as exc:
            return Response(
                {
                    "error": str(exc)
                },
                status=status.HTTP_400_BAD_REQUEST,
            )