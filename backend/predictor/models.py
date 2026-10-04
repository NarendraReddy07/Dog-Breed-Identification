from django.db import models


class PredictionLog(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)

    filename = models.CharField(
        max_length=255,
        blank=True,
    )

    predicted_breed = models.CharField(
        max_length=120,
    )

    confidence = models.FloatField()

    latency_ms = models.FloatField()

    model_version = models.CharField(
        max_length=100,
        default="tf-ensemble-v1",
    )

    def __str__(self):
        return (
            f"{self.predicted_breed} "
            f"({self.confidence:.2%})"
        )