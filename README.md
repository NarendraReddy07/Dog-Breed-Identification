# Dog Breed Identification AI

A full-stack deep learning application that predicts a dog's breed from an uploaded image. The project combines a React frontend, Django REST API, and TensorFlow/Keras inference pipeline to classify images across 120 dog breeds.

## Features

- **120-breed classification** with top-five predictions and confidence scores.
- **Deep learning inference** using ViT-L/32, EfficientNetB3, and InceptionResNetV2 feature extraction with five trained stack classifier heads.
- **Production configuration** with deterministic inference and temperature calibration.
- **Image upload interface** with preview, file validation, and prediction status.
- **Prediction analytics** showing total predictions, average confidence, average latency, model version, and frequently predicted breeds.
- **REST API** built with Django REST Framework.
- **Automated tests** for frontend behavior and backend functionality.
- **Docker support** for the application environment.

## Model Performance

The following metrics are recorded in the model's validation and production configuration files.

| Metric | Result |
|---|---:|
| Number of classes | 120 |
| Validation samples | 1,023 |
| Top-1 validation accuracy | 94.72% |
| Top-5 validation accuracy | 99.51% |
| Calibrated validation log loss | 0.19564 |
| Production branch | Stack |
| Temperature | 1.10 |

These are validation results, not guaranteed accuracy on unseen real-world images. Prediction confidence is the model's estimated probability and should not be interpreted as a guarantee of correctness.

## Architecture

```text
React + Vite Frontend
        |
        | HTTP requests
        v
Django REST Framework API
        |
        +---- Image validation and preprocessing
        |
        v
Feature Extraction
  +----------------------+
  | ViT-L/32             |
  | EfficientNetB3       |
  | InceptionResNetV2    |
  +----------------------+
        |
        v
Five Stack Classifier Heads
        |
        v
Probability Averaging
        |
        v
Temperature Calibration
        |
        v
Top-5 Breed Predictions
        |
        +---- Prediction History
        |
        +---- Analytics Dashboard
```

### Technology Stack

- **Frontend:** React, Vite, JavaScript, CSS
- **Backend:** Python, Django, Django REST Framework
- **Machine Learning:** TensorFlow, Keras, KerasHub, NumPy
- **Image Processing:** Pillow
- **Testing:** Vitest, React Testing Library, Django tests
- **Development Tools:** Git, GitHub, Docker, VS Code

## Project Structure

```text
Dog-Breed-Identification/
├── artifacts/
│   ├── production_config.json
│   ├── class_mapping.json
│   ├── ensemble_calibration.json
│   ├── validation_metrics.json
│   └── stack_seed_*.keras
├── backend/
│   ├── config/
│   ├── predictor/
│   │   ├── model_manager.py
│   │   ├── views.py
│   │   ├── models.py
│   │   └── tests.py
│   ├── manage.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── App.test.jsx
│   ├── package.json
│   └── Dockerfile
├── ml/
│   ├── train.py
│   ├── evaluate.py
│   ├── ensemble.py
│   ├── calibration.py
│   └── export.py
├── docker-compose.yml
└── README.md
```

## Run Locally

### Prerequisites

- Python and a compatible virtual environment
- Node.js and npm
- Git
- Sufficient memory and disk space for the machine learning dependencies and model artifacts

### 1. Clone the repository

```bash
git clone https://github.com/NarendraReddy07/Dog-Breed-Identification.git
cd Dog-Breed-Identification
```

### 2. Start the backend

Activate your Python virtual environment, install the backend dependencies from `backend/requirements.txt` if necessary, and run:

```bash
cd backend
python manage.py migrate
python manage.py runserver
```

The backend runs at `http://127.0.0.1:8000/` by default.

The model artifacts must be present in the root `artifacts/` directory. The first prediction may take longer while the backbones and classifier heads load.

### 3. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite. The default is usually `http://localhost:5173/`, with the repository's configured base path applied where relevant.

Keep the backend and frontend terminals running while using the application.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/predict/` | Predict the breed from an uploaded image |
| GET | `/api/analytics/` | Retrieve prediction analytics |
| GET | `/api/history/` | Retrieve recent prediction history |

The prediction endpoint expects an image uploaded using the multipart form field `image`. Supported formats are JPEG, PNG, and WEBP, with a maximum upload size of 10 MB.

## Testing

### Frontend

Run from the `frontend` directory:

```bash
npm test -- --run
npm run lint
npm run build
```

Verified results:

- 12 frontend tests passed.
- Lint completed with zero warnings and zero errors.
- The Vite production build completed successfully.

### Backend

Run from the `backend` directory:

```bash
python manage.py test
```

Verified result: six Django tests passed, with no failures.

These results describe the tested local code at the time of verification.

## Model Artifacts and Data

The production configuration identifies the selected inference branch as `stack`, with a stack weight of `1.0`, a temperature of `1.1`, an input image size of 384 × 384, and 120 output classes.

The production package includes five stack classifier heads, the class mapping, calibration configuration, production configuration, and validation metrics.

Dataset files and trained model weights may be excluded from version control because of their size. Follow the project's data and artifact setup requirements when preparing a fresh environment. Do not commit secrets, environment files, local databases, or large data files unintentionally.

## Limitations

- The model predicts among the 120 supported classes; it is not a general-purpose animal identification system.
- Validation accuracy does not guarantee correct predictions for every image.
- Confidence scores may be high even when a prediction is incorrect.
- CPU inference can be slower than inference on supported GPU hardware.
- The analytics dashboard summarizes predictions recorded by the local backend database.
- The project is currently verified for local use. Public deployment is not part of the current setup.

## Future Improvements

- Add a broader labeled-image evaluation suite.
- Expand API integration and failure-path testing.
- Improve inference throughput where needed.
- Add a public demo when a suitable hosting option is available.

## License

Add a license file before distributing the project under an explicit open-source license.