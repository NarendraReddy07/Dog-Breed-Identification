import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import axios from "axios";
import App from "./App";

vi.mock("axios");

describe("Dog Breed Identifier", () => {
  beforeEach(() => {
    localStorage.clear();

    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:test-image"),
    });

    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn(),
    });

    axios.get.mockResolvedValue({
      data: {
        total_predictions: 3,
        average_confidence: 0.9672,
        average_latency_ms: 42533,
        top_breeds: [
          {
            predicted_breed: "golden_retriever",
            count: 2,
          },
          {
            predicted_breed: "labrador_retriever",
            count: 1,
          },
        ],
        model_versions: [
          {
            model_version: "tf-ensemble-v1",
            count: 3,
          },
        ],
      },
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders the application title", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", {
        name: /dog breed identifier/i,
      })
    ).toBeInTheDocument();
  });

  it("renders the upload interface", () => {
    render(<App />);

    expect(
      screen.getByText(/drop your dog image here/i)
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        /or click to browse from your computer/i
      )
    ).toBeInTheDocument();

    expect(
      screen.getByRole("button", {
        name: /predict breed/i,
      })
    ).toBeDisabled();
  });

  it("renders the theme toggle", () => {
    render(<App />);

    expect(
      screen.getByRole("button", {
        name: /switch to/i,
      })
    ).toBeInTheDocument();
  });

  it("renders analytics metrics and popular breeds", async () => {
    render(<App />);

    expect(
      await screen.findByText("Total Predictions")
    ).toBeInTheDocument();

    expect(
      screen.getByText("Prediction Overview")
    ).toBeInTheDocument();

    expect(
      screen.getByText("Average Confidence")
    ).toBeInTheDocument();

    expect(
      screen.getByText("Average Latency")
    ).toBeInTheDocument();

    expect(
      screen.getByText("Model Version")
    ).toBeInTheDocument();

    expect(
      screen.getByText("Most Predicted Breeds")
    ).toBeInTheDocument();

    expect(
      screen.getByText("Golden Retriever")
    ).toBeInTheDocument();

    expect(
      screen.getByText("Labrador Retriever")
    ).toBeInTheDocument();
  });

  it("shows an analytics error when the API request fails", async () => {
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});

    axios.get.mockRejectedValueOnce(
      new Error("Network error")
    );

    render(<App />);

    expect(
      await screen.findByText(
        "Analytics are currently unavailable."
      )
    ).toBeInTheDocument();

    consoleError.mockRestore();
  });

  it("uploads an image and displays its preview", async () => {
    const { container } = render(<App />);

    const file = new File(
      ["fake image"],
      "dog.jpg",
      { type: "image/jpeg" }
    );

    const fileInput = container.querySelector(
      'input[type="file"]'
    );

    expect(fileInput).not.toBeNull();

    fireEvent.change(fileInput, {
      target: { files: [file] },
    });

    expect(
      await screen.findByAltText("Selected dog")
    ).toBeInTheDocument();

    expect(
      screen.getByText("dog.jpg")
    ).toBeInTheDocument();

    expect(
      screen.getByRole("button", {
        name: /predict breed/i,
      })
    ).toBeEnabled();
  });

  it("displays the prediction returned by the API", async () => {
    axios.post.mockResolvedValueOnce({
      data: {
        top_prediction: {
          breed: "golden_retriever",
          confidence: 0.985,
        },
        predictions: [
          {
            breed: "golden_retriever",
            confidence: 0.985,
          },
          {
            breed: "labrador_retriever",
            confidence: 0.01,
          },
        ],
        model_version: "tf-ensemble-v1",
        image: {
          width: 224,
          height: 224,
        },
        deterministic_inference: true,
        latency_ms: 2500,
      },
    });

    const { container } = render(<App />);

    const file = new File(
      ["fake image"],
      "dog.jpg",
      { type: "image/jpeg" }
    );

    const fileInput = container.querySelector(
      'input[type="file"]'
    );

    expect(fileInput).not.toBeNull();

    fireEvent.change(fileInput, {
      target: { files: [file] },
    });

    const predictButton = await screen.findByRole(
      "button",
      { name: /predict breed/i }
    );

    await waitFor(() => {
      expect(predictButton).toBeEnabled();
    });

    fireEvent.click(predictButton);

    expect(
      await screen.findByRole("heading", {
        name: "We found a match",
      })
    ).toBeInTheDocument();

    expect(
      screen.getByRole("heading", {
        name: "Golden Retriever",
      })
    ).toBeInTheDocument();

    expect(
      screen.getByText("98.50%")
    ).toBeInTheDocument();

    expect(axios.post).toHaveBeenCalledWith(
      "/api/predict/",
      expect.any(FormData)
    );
  });

  it("opens the image picker when the upload area receives Enter", () => {
    render(<App />);

    const uploadArea = screen.getByRole("button", {
      name: "Choose a dog image",
    });

    const fileInput = document.querySelector(
      'input[type="file"]'
    );

    const clickSpy = vi
      .spyOn(fileInput, "click")
      .mockImplementation(() => {});

    fireEvent.keyDown(uploadArea, {
      key: "Enter",
    });

    expect(clickSpy).toHaveBeenCalledOnce();

    clickSpy.mockRestore();
  });
});