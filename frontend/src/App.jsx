import { useEffect, useRef, useState } from "react";
import axios from "axios";
import {
  Upload,
  Image as ImageIcon,
  Sparkles,
  RotateCcw,
  AlertCircle,
  Sun,
  Moon,
  BarChart3,
  Activity,
  Clock3,
  Database,
} from "lucide-react";
import "./App.css";

function formatBreedName(breed) {
  return breed
    .split("-")
    .map(
      (word) =>
        word.charAt(0).toUpperCase() + word.slice(1)
    )
    .join(" ");
}

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    return (
      localStorage.getItem("dog-breed-theme") === "dark"
    );
  });

  useEffect(() => {
    document.documentElement.dataset.theme = darkMode
      ? "dark"
      : "light";

    localStorage.setItem(
      "dog-breed-theme",
      darkMode ? "dark" : "light"
    );
  }, [darkMode]);

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [analyticsLoading, setAnalyticsLoading] =
    useState(true);
  const [analyticsError, setAnalyticsError] =
    useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const fileInputRef = useRef(null);

  const loadAnalytics = async () => {
    try {
      setAnalyticsLoading(true);
      setAnalyticsError("");

      const response = await axios.get(
        "/api/analytics/"
      );

      setAnalytics(response.data);
    } catch (requestError) {
      console.error(requestError);

      setAnalyticsError(
        "Analytics are currently unavailable."
      );
    } finally {
      setAnalyticsLoading(false);
    }
  };

  useEffect(() => {
    loadAnalytics();
  }, []);

  const handleFile = (file) => {
    if (!file) {
      return;
    }

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError("");
  };

  const handleFileInput = (event) => {
    const file = event.target.files?.[0];
    handleFile(file);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);

    const file = event.dataTransfer.files?.[0];
    handleFile(file);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const predictBreed = async () => {
    if (!selectedFile) {
      setError("Please select a dog image first.");
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("image", selectedFile);

      const response = await axios.post(
        "/api/predict/",
        formData
      );

      setResult(response.data);

      await loadAnalytics();
    } catch (requestError) {
      console.error(requestError);

      const message =
        requestError.response?.data?.error ||
        "Prediction failed. Make sure the Django backend is running.";

      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const resetApp = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const averageConfidence =
    (analytics?.average_confidence ?? 0) * 100;

  const averageLatency =
    (analytics?.average_latency_ms ?? 0) / 1000;

  return (
    <div className="app">
      <header className="hero">
        <button
          className="theme-toggle"
          onClick={() =>
            setDarkMode((current) => !current)
          }
          aria-label={
            darkMode
              ? "Switch to light mode"
              : "Switch to dark mode"
          }
          title={
            darkMode
              ? "Switch to light mode"
              : "Switch to dark mode"
          }
        >
          {darkMode ? (
            <Sun size={20} />
          ) : (
            <Moon size={20} />
          )}
        </button>

        <div className="hero-badge">
          <Sparkles size={16} />
          TensorFlow + Django + React
        </div>

        <h1>Dog Breed Identifier</h1>

        <p>
          Upload a dog image and let the AI identify its
          breed using a 120-class deep learning model.
        </p>
      </header>

      <main className="container">
        <section className="upload-card">
          <div
            className={`drop-zone ${
              isDragging ? "dragging" : ""
            } ${previewUrl ? "has-preview" : ""}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() =>
              fileInputRef.current?.click()
            }
          >
            {previewUrl ? (
              <div className="preview-wrapper">
                <img
                  src={previewUrl}
                  alt="Selected dog"
                  className="preview-image"
                />

                <div className="preview-overlay">
                  <ImageIcon size={20} />
                  <span>
                    Click to choose another image
                  </span>
                </div>
              </div>
            ) : (
              <div className="upload-content">
                <div className="upload-icon">
                  <Upload size={32} />
                </div>

                <h2>Drop your dog image here</h2>

                <p>
                  or click to browse from your computer
                </p>

                <span className="file-hint">
                  JPG, JPEG, PNG or WEBP
                </span>
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              hidden
            />
          </div>

          {selectedFile && (
            <div className="selected-file">
              <ImageIcon size={18} />

              <div>
                <strong>{selectedFile.name}</strong>
                <span>
                  {(
                    selectedFile.size /
                    1024 /
                    1024
                  ).toFixed(2)}{" "}
                  MB
                </span>
              </div>
            </div>
          )}

          <div className="action-row">
            <button
              className="predict-button"
              onClick={predictBreed}
              disabled={!selectedFile || isLoading}
            >
              <Sparkles size={20} />

              {isLoading
                ? "Analyzing..."
                : "Predict Breed"}
            </button>

            {(selectedFile || result) && (
              <button
                className="reset-button"
                onClick={resetApp}
              >
                <RotateCcw size={18} />
                Reset
              </button>
            )}
          </div>

          {isLoading && (
            <div className="loading-box">
              <div className="spinner"></div>

              <div>
                <strong>
                  Analyzing image...
                </strong>
                <span>
                  The AI models are processing your
                  image.
                </span>
              </div>
            </div>
          )}

          {error && (
            <div className="error-box">
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}
        </section>

        {result && (
          <section className="results-card">
            <div className="results-header">
              <div>
                <span className="section-label">
                  Prediction Result
                </span>

                <h2>We found a match</h2>
              </div>

              <span className="model-version">
                {result.model_version}
              </span>
            </div>

            <div className="top-prediction">
              <div>
                <span className="prediction-label">
                  Top Prediction
                </span>

                <h3>
                  {formatBreedName(
                    result.top_prediction.breed
                  )}
                </h3>
              </div>

              <div className="top-confidence">
                {(
                  result.top_prediction.confidence *
                  100
                ).toFixed(2)}
                %
              </div>
            </div>

            <div className="top-five">
              <div className="section-heading">
                <h3>Top 5 Predictions</h3>
                <span>Confidence</span>
              </div>

              {result.predictions.map(
                (prediction, index) => {
                  const percentage =
                    prediction.confidence * 100;

                  return (
                    <div
                      className="prediction-row"
                      key={prediction.breed}
                    >
                      <div className="prediction-info">
                        <span className="rank">
                          {index + 1}
                        </span>

                        <span className="breed-name">
                          {formatBreedName(
                            prediction.breed
                          )}
                        </span>

                        <span className="percentage">
                          {percentage.toFixed(2)}%
                        </span>
                      </div>

                      <div className="progress-track">
                        <div
                          className="progress-fill"
                          style={{
                            width: `${Math.max(
                              percentage,
                              0.5
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                  );
                }
              )}
            </div>

            <div className="metadata">
              <div>
                <span>Image</span>
                <strong>
                  {result.image.width} ×{" "}
                  {result.image.height}
                </strong>
              </div>

              <div>
                <span>Inference</span>
                <strong>
                  {result.deterministic_inference
                    ? "Deterministic"
                    : "Non-deterministic"}
                </strong>
              </div>

              <div>
                <span>Latency</span>
                <strong>
                  {(result.latency_ms / 1000).toFixed(
                    2
                  )}
                  s
                </strong>
              </div>
            </div>
          </section>
        )}

        <section className="analytics-section">
          <div className="analytics-header">
            <div>
              <span className="section-label">
                Model Analytics
              </span>

              <h2>Prediction Overview</h2>

              <p>
                Live metrics calculated from recorded
                predictions.
              </p>
            </div>

            <BarChart3 size={28} />
          </div>

          {analyticsLoading ? (
            <div className="analytics-loading">
              <div className="spinner"></div>
              <span>
                Loading analytics...
              </span>
            </div>
          ) : analyticsError ? (
            <div className="error-box">
              <AlertCircle size={20} />
              <span>{analyticsError}</span>
            </div>
          ) : (
            <>
              <div className="analytics-grid">
                <div className="metric-card">
                  <div className="metric-icon">
                    <Database size={20} />
                  </div>

                  <span>Total Predictions</span>

                  <strong>
                    {analytics.total_predictions}
                  </strong>
                </div>

                <div className="metric-card">
                  <div className="metric-icon">
                    <Sparkles size={20} />
                  </div>

                  <span>Average Confidence</span>

                  <strong>
                    {averageConfidence.toFixed(2)}%
                  </strong>
                </div>

                <div className="metric-card">
                  <div className="metric-icon">
                    <Clock3 size={20} />
                  </div>

                  <span>Average Latency</span>

                  <strong>
                    {averageLatency.toFixed(2)}s
                  </strong>
                </div>

                <div className="metric-card">
                  <div className="metric-icon">
                    <Activity size={20} />
                  </div>

                  <span>Model Version</span>

                  <strong>
                    {analytics.model_versions?.[0]
                      ?.model_version ||
                      "N/A"}
                  </strong>
                </div>
              </div>

              <div className="analytics-breeds">
                <div className="section-heading">
                  <h3>Most Predicted Breeds</h3>
                  <span>Recorded predictions</span>
                </div>

                {analytics.top_breeds?.length ? (
                  analytics.top_breeds
                    .slice(0, 5)
                    .map((breed, index) => (
                      <div
                        className="analytics-breed-row"
                        key={breed.predicted_breed}
                      >
                        <div className="analytics-rank">
                          {index + 1}
                        </div>

                        <div className="analytics-breed-name">
                          {formatBreedName(
                            breed.predicted_breed
                          )}
                        </div>

                        <div className="analytics-count">
                          {breed.count}{" "}
                          {breed.count === 1
                            ? "prediction"
                            : "predictions"}
                        </div>
                      </div>
                    ))
                ) : (
                  <p className="analytics-empty">
                    No predictions have been recorded
                    yet.
                  </p>
                )}
              </div>
            </>
          )}
        </section>
      </main>

      <footer>
        <span>
          Dog Breed AI · 120 breeds · TensorFlow
        </span>
      </footer>
    </div>
  );
}

export default App;