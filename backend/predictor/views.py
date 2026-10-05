import logging
import time
from io import BytesIO

import tensorflow as tf
from django.db.models import Avg, Count
from PIL import Image, UnidentifiedImageError
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .model_manager import model_manager
from .models import PredictionLog


logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


class PredictView(APIView):
    """
    Predict the breed of an uploaded dog image
    and store the prediction in the database.
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

        if uploaded_file.content_type not in ALLOWED_IMAGE_TYPES:
            return Response(
                {
                    "error": (
                        "Unsupported image type. "
                        "Use JPEG, PNG, or WEBP."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        if uploaded_file.size > 10 * 1024 * 1024:
            return Response(
                {
                    "error": "Image file is too large. Maximum size is 10 MB."
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

            top_prediction = predictions[0]

            PredictionLog.objects.create(
                filename=uploaded_file.name,
                predicted_breed=top_prediction["breed"],
                confidence=top_prediction["confidence"],
                latency_ms=latency_ms,
                model_version="tf-ensemble-v1",
            )

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
                    "top_prediction": top_prediction,
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

        except (UnidentifiedImageError, OSError):
            logger.warning(
                "Invalid image upload: %s",
                uploaded_file.name,
                exc_info=True,
            )

            return Response(
                {
                    "error": "The uploaded file is not a valid image."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        except Exception:
            logger.exception(
                "Prediction failed for uploaded file: %s",
                uploaded_file.name,
            )

            return Response(
                {
                    "error": (
                        "Prediction failed due to an internal "
                        "server error."
                    )
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class PredictionHistoryView(APIView):
    """
    Return recent prediction history.
    """

    def get(self, request):
        logs = PredictionLog.objects.order_by(
            "-timestamp"
        )[:100]

        data = [
            {
                "id": log.id,
                "timestamp": log.timestamp,
                "filename": log.filename,
                "predicted_breed": log.predicted_breed,
                "confidence": log.confidence,
                "latency_ms": log.latency_ms,
                "model_version": log.model_version,
            }
            for log in logs
        ]

        return Response(
            {
                "count": len(data),
                "predictions": data,
            },
            status=status.HTTP_200_OK,
        )


class PredictionAnalyticsView(APIView):
    """
    Return aggregate prediction metrics.
    """

    def get(self, request):
        total_predictions = PredictionLog.objects.count()

        aggregates = PredictionLog.objects.aggregate(
            average_confidence=Avg("confidence"),
            average_latency_ms=Avg("latency_ms"),
        )

        top_breeds = (
            PredictionLog.objects
            .values("predicted_breed")
            .annotate(count=Count("id"))
            .order_by("-count", "predicted_breed")[:10]
        )

        model_versions = (
            PredictionLog.objects
            .values("model_version")
            .annotate(count=Count("id"))
            .order_by("-count", "model_version")
        )

        return Response(
            {
                "total_predictions": total_predictions,
                "average_confidence": (
                    aggregates["average_confidence"]
                    if aggregates["average_confidence"] is not None
                    else 0
                ),
                "average_latency_ms": (
                    aggregates["average_latency_ms"]
                    if aggregates["average_latency_ms"] is not None
                    else 0
                ),
                "top_breeds": list(top_breeds),
                "model_versions": list(model_versions),
            },
            status=status.HTTP_200_OK,
        )