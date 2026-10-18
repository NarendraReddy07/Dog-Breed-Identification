# Dog Breed Identification

A full-stack deep learning application that predicts a dog's breed from an uploaded image. The project combines a React interface, a Django REST API, and TensorFlow/Keras models to return breed predictions and confidence scores.

## Features

- **120-breed classification:** Identifies breeds supported by the trained model.
- **Image upload:** Select an image from your computer or drag it into the upload area.
- **Top predictions:** View the leading breed predictions and their confidence scores.
- **Prediction metadata:** Inspect image dimensions, inference mode, and prediction latency.
- **Model analytics:** View recorded prediction counts, average confidence, latency, and frequently predicted breeds.
- **Light and dark themes:** Switch between display themes.
- **Docker support:** Run the application components in a local containerized environment.
- **Automated frontend tests:** Test the interface, upload preview, prediction flow, analytics, and error handling.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, JavaScript, Vite |
| UI icons | Lucide React |
| API communication | Axios |
| Backend | Python, Django REST Framework |
| Machine learning | TensorFlow / Keras |
| Model architecture | Ensemble-based classification |
| Containerization | Docker |
| Frontend testing | Vitest, React Testing Library |
| Source control | Git and GitHub |

## Architecture

```text
User
 |
 v
React + Vite frontend
 |
 | HTTP requests
 v
Django REST API
 |
 +----------------------+
 |                      |
 v                      v
Prediction endpoint     Analytics endpoint
 |
 v
TensorFlow / Keras inference
 |
 v
Ranked breed predictions
 |
 v
Results displayed in the frontend
```

## How It Works

1. The user selects a dog image.
2. The frontend displays a preview and allows the user to request a prediction.
3. The image is submitted to the Django API as multipart form data.
4. The backend processes the image using the available trained models.
5. The API returns the predicted breed, confidence scores, and inference metadata.
6. The frontend displays the results and refreshes the analytics.

The classifier supports 120 breed classes. Its real-world performance depends on the training data, image quality, and the characteristics of the uploaded image.

## Project Structure

```text
Dog-Breed-Identification/
├── backend/
│   ├── manage.py
│   ├── Dockerfile
│   └── Django API, settings, and model inference code
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── App.test.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── Dockerfile
│   ├── package.json
│   └── vite.config.js
├── docker-compose.yml
└── README.md
```

Some model files, datasets, and generated runtime data may be excluded from version control because of their size or local configuration requirements.

## Run the Frontend Locally

### Requirements

- Node.js and npm
- The project repository cloned to your computer

### Setup

Open a terminal in the repository's `frontend` directory:

```bash
npm install
npm run dev
```

Vite will print the local development URL in the terminal.

### Run the Tests

```bash
npm test -- --run
```

### Create a Production Build

```bash
npm run build
```

The generated frontend build is written to the `frontend/dist/` directory.

### Run Lint Checks

```bash
npm run lint
```

## Backend and API

The frontend communicates with these API routes:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/analytics/` | Retrieves prediction analytics |
| POST | `/api/predict/` | Submits an image for breed classification |

The prediction request uses multipart form data with the image field named `image`.

For local integration, start the Django backend using the configuration and dependencies included in the repository. If the API is hosted at a different address, configure the frontend's `VITE_API_BASE_URL` environment variable accordingly.

The backend must be running for real predictions and live analytics. Frontend tests use mocked API responses and do not require model inference.

## Dataset and Model Files

The project uses a dog-breed dataset for training and evaluation. Dataset files and trained model weights are not included in the Git repository.

Check the repository's existing data and model setup instructions before running training or inference on a fresh machine. Do not commit large datasets, model weights, local databases, secrets, or environment files accidentally.

## Testing and Verification

The frontend currently contains **12 automated tests** covering:

- Application title and upload interface rendering.
- Theme toggle and analytics display.
- Analytics API error handling.
- Image upload and preview rendering.
- Rejection of unsupported file types.
- Rejection of images larger than 10 MB.
- Display of predictions returned by the API.
- Resetting the selected image and prediction.
- Prediction API failure handling and loading-state cleanup.
- Keyboard accessibility for opening the image picker.

Run the frontend tests from the `frontend` directory:

```bash
npm test -- --run
```

Run the linter:

```bash
npm run lint
```

Build the production frontend:

```bash
npm run build
```

The current frontend verification results are 12 passing tests, zero lint warnings or errors, and a successful Vite production build.

These checks validate the tested frontend behavior. They do not, by themselves, establish model accuracy on unseen data or guarantee that a remote backend is available.

## Current Limitations

- Live predictions require a reachable Django backend with the required model files and dependencies.
- Prediction latency depends on the available hardware and model initialization.
- Confidence scores represent the model's output and should not be interpreted as a guarantee of correctness.
- Public hosting and remote backend deployment are not included in the current project scope.

## Future Improvements

- Expand automated backend and API integration tests.
- Improve accessibility and keyboard navigation.
- Add reproducible model evaluation reports.
- Improve inference efficiency and monitoring.
- Add example screenshots and measured evaluation metrics.

## License

Add a license file if you intend to distribute the project under a specific open-source license.