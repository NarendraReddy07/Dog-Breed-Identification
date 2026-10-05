import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import axios from "axios";
import App from "./App";

vi.mock("axios");

describe("Dog Breed Identifier", () => {
  beforeEach(() => {
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
      screen.getByText(/or click to browse from your computer/i)
    ).toBeInTheDocument();

    expect(
      screen.getByRole("button", {
        name: /predict breed/i,
      })
    ).toBeInTheDocument();
  });

  it("renders the theme toggle", () => {
    render(<App />);

    expect(
      screen.getByRole("button", {
        name: /switch to/i,
      })
    ).toBeInTheDocument();
  });

  it("renders model analytics", async () => {
    render(<App />);

    await waitFor(() => {
      expect(
        screen.getByText("Model Analytics")
      ).toBeInTheDocument();
    });

    expect(
      screen.getByText("Prediction Overview")
    ).toBeInTheDocument();

    expect(
      screen.getByText("Total Predictions")
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
      screen.getByText("Golden_retriever")
    ).toBeInTheDocument();
  });
});