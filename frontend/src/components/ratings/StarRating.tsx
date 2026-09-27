"use client";

import React, { useState } from "react";
import { Star } from "lucide-react";

interface StarRatingProps {
  value: number;
  onChange?: (val: number) => void;
  readonly?: boolean;
  size?: "sm" | "md" | "lg";
}

export const StarRating: React.FC<StarRatingProps> = ({
  value,
  onChange,
  readonly = false,
  size = "md",
}) => {
  const [hoverVal, setHoverVal] = useState<number | null>(null);

  const activeValue = hoverVal !== null ? hoverVal : value;

  const starSizes = {
    sm: "w-4 h-4",
    md: "w-5 h-5",
    lg: "w-6 h-6",
  };

  const handleStarClick = (starIndex: number, isHalf: boolean) => {
    if (readonly || !onChange) return;
    const newRating = starIndex + (isHalf ? 0.5 : 1.0);
    onChange(newRating);
  };

  const handleMouseMove = (
    e: React.MouseEvent<HTMLDivElement>,
    starIndex: number
  ) => {
    if (readonly) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const isHalf = e.clientX - rect.left < rect.width / 2;
    setHoverVal(starIndex + (isHalf ? 0.5 : 1.0));
  };

  return (
    <div
      className="flex items-center gap-1"
      onMouseLeave={() => !readonly && setHoverVal(null)}
    >
      {[0, 1, 2, 3, 4].map((starIndex) => {
        const fillAmount = Math.max(
          0,
          Math.min(1, activeValue - starIndex)
        );

        return (
          <div
            key={starIndex}
            className={`relative cursor-pointer transition-transform ${
              readonly ? "cursor-default" : "hover:scale-110"
            }`}
            onMouseMove={(e) => handleMouseMove(e, starIndex)}
            onClick={(e) => {
              if (readonly || !onChange) return;
              const rect = e.currentTarget.getBoundingClientRect();
              const isHalf = e.clientX - rect.left < rect.width / 2;
              handleStarClick(starIndex, isHalf);
            }}
          >
            {/* Background empty star */}
            <Star
              className={`${starSizes[size]} text-slate-700 fill-slate-800`}
            />

            {/* Filled star portion (half or full) */}
            {fillAmount > 0 && (
              <div
                className="absolute top-0 left-0 overflow-hidden pointer-events-none"
                style={{ width: `${fillAmount * 100}%` }}
              >
                <Star
                  className={`${starSizes[size]} text-amber-400 fill-amber-400`}
                />
              </div>
            )}
          </div>
        );
      })}
      {value > 0 && (
        <span className="ml-1.5 font-bold text-slate-300 text-xs">
          {value.toFixed(1)}
        </span>
      )}
    </div>
  );
};
