import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("Dog Breed Identifier", () => {
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
});