from django.contrib import admin

from .models import PredictionLog


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = (
        "timestamp",
        "predicted_breed",
        "confidence",
        "latency_ms",
        "model_version",
        "filename",
    )

    list_filter = (
        "predicted_breed",
        "model_version",
    )

    search_fields = (
        "predicted_breed",
        "filename",
    )

    ordering = (
        "-timestamp",
    )