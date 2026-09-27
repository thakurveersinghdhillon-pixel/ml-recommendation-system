import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { StarRating } from "../components/ratings/StarRating";
import { SkeletonCard } from "../components/ui/SkeletonCard";
import { MovieCard } from "../components/movies/MovieCard";

describe("Frontend Component Suite", () => {
  it("StarRating renders correct active value and triggers onChange", () => {
    const handleChange = vi.fn();
    render(<StarRating value={3.5} onChange={handleChange} />);
    expect(screen.getByText("3.5")).toBeInTheDocument();
  });

  it("SkeletonCard renders pulse animation element", () => {
    const { container } = render(<SkeletonCard />);
    expect(container.firstChild).toHaveClass("animate-pulse");
  });

  it("MovieCard displays movie details and title link", () => {
    const mockMovie = {
      movie_id: 1,
      title: "Toy Story (1995)",
      genres: ["Adventure", "Animation"],
      year: 1995,
    };
    render(<MovieCard movie={mockMovie} />);
    expect(screen.getByText("Toy Story (1995)")).toBeInTheDocument();
    expect(screen.getByText("Adventure")).toBeInTheDocument();
    expect(screen.getByText("Animation")).toBeInTheDocument();
  });
});
