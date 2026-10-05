from io import BytesIO
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from PIL import Image
from rest_framework.test import APIClient

from predictor.models import PredictionLog


def create_test_image():
    """
    Create a small valid JPEG upload entirely in memory.
    """
    image = Image.new(
        "RGB",
        (200, 150),
        color=(120, 120, 120),
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
    )

    return SimpleUploadedFile(
        "test.jpg",
        buffer.getvalue(),
        content_type="image/jpeg",
    )


class PredictAPITest(TestCase):

    def setUp(self):
        self.client = APIClient()

    def test_predict_without_image_returns_400(self):
        response = self.client.post(
            "/api/predict/",
        )

        self.assertEqual(
            response.status_code,
            400,
        )

        self.assertIn(
            "error",
            response.data,
        )

    @patch(
        "predictor.views.model_manager.predict"
    )
    def test_predict_with_valid_image(self, mock_predict):
        mock_predict.return_value = [
            {
                "breed": "golden_retriever",
                "confidence": 0.91,
            },
            {
                "breed": "labrador_retriever",
                "confidence": 0.04,
            },
            {
                "breed": "german_shepherd",
                "confidence": 0.02,
            },
            {
                "breed": "golden_retriever",
                "confidence": 0.02,
            },
            {
                "breed": "beagle",
                "confidence": 0.01,
            },
        ]

        image = create_test_image()

        response = self.client.post(
            "/api/predict/",
            {
                "image": image,
            },
            format="multipart",
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertIn(
            "predictions",
            response.data,
        )

        self.assertIn(
            "top_prediction",
            response.data,
        )

        self.assertIn(
            "model_version",
            response.data,
        )

        self.assertEqual(
            len(response.data["predictions"]),
            5,
        )

        self.assertEqual(
            response.data["top_prediction"]["breed"],
            "golden_retriever",
        )

        mock_predict.assert_called_once()

    def test_predict_rejects_non_image(self):
        text_file = SimpleUploadedFile(
            "test.txt",
            b"This is not an image.",
            content_type="text/plain",
        )

        response = self.client.post(
            "/api/predict/",
            {
                "image": text_file,
            },
            format="multipart",
        )

        self.assertEqual(
            response.status_code,
            400,
        )

        self.assertEqual(
            response.data["error"],
            "Unsupported image type. Use JPEG, PNG, or WEBP.",
        )

    def test_predict_rejects_large_file(self):
        large_file = SimpleUploadedFile(
            "large.jpg",
            b"x" * (10 * 1024 * 1024 + 1),
            content_type="image/jpeg",
        )
    
        response = self.client.post(
            "/api/predict/",
            {
                "image": large_file,
            },
            format="multipart",
        )
    
        self.assertEqual(
            response.status_code,
            400,
        )
    
        self.assertEqual(
            response.data["error"],
            "Image file is too large. Maximum size is 10 MB.",
        )

    def test_prediction_history_returns_saved_predictions(self):
        PredictionLog.objects.create(
            filename="test-dog.jpg",
            predicted_breed="golden_retriever",
            confidence=0.95,
            latency_ms=1234.56,
            model_version="tf-ensemble-v1",
        )

        response = self.client.get(
            "/api/predictions/"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertEqual(
            response.data["count"],
            1,
        )

        self.assertEqual(
            len(response.data["predictions"]),
            1,
        )

        prediction = response.data["predictions"][0]

        self.assertEqual(
            prediction["filename"],
            "test-dog.jpg",
        )

        self.assertEqual(
            prediction["predicted_breed"],
            "golden_retriever",
        )

        self.assertEqual(
            prediction["model_version"],
            "tf-ensemble-v1",
        )

    def test_prediction_analytics_returns_metrics(self):
        PredictionLog.objects.create(
            filename="dog-1.jpg",
            predicted_breed="golden_retriever",
            confidence=0.90,
            latency_ms=1000.0,
            model_version="tf-ensemble-v1",
        )

        PredictionLog.objects.create(
            filename="dog-2.jpg",
            predicted_breed="golden_retriever",
            confidence=0.80,
            latency_ms=2000.0,
            model_version="tf-ensemble-v1",
        )

        PredictionLog.objects.create(
            filename="dog-3.jpg",
            predicted_breed="beagle",
            confidence=0.70,
            latency_ms=3000.0,
            model_version="tf-ensemble-v1",
        )

        response = self.client.get(
            "/api/analytics/"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertEqual(
            response.data["total_predictions"],
            3,
        )

        self.assertAlmostEqual(
            response.data["average_confidence"],
            0.80,
            places=5,
        )

        self.assertAlmostEqual(
            response.data["average_latency_ms"],
            2000.0,
            places=5,
        )

        self.assertEqual(
            response.data["top_breeds"][0]["predicted_breed"],
            "golden_retriever",
        )

        self.assertEqual(
            response.data["top_breeds"][0]["count"],
            2,
        )

        self.assertEqual(
            response.data["model_versions"][0]["model_version"],
            "tf-ensemble-v1",
        )

        self.assertEqual(
            response.data["model_versions"][0]["count"],
            3,
        )