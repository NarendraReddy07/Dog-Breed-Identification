from django.urls import path

from .views import (
    PredictView,
    PredictionAnalyticsView,
    PredictionHistoryView,
)


urlpatterns = [
    path(
        "predict/",
        PredictView.as_view(),
        name="predict",
    ),
    path(
        "predictions/",
        PredictionHistoryView.as_view(),
        name="prediction-history",
    ),
    path(
        "analytics/",
        PredictionAnalyticsView.as_view(),
        name="prediction-analytics",
    ),
]