import React from "react";
import { Film } from "lucide-react";

export const Footer: React.FC = () => {
  return (
    <footer className="bg-slate-950 border-t border-slate-800/80 py-8 text-slate-400 text-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <Film className="w-4 h-4 text-amber-500" />
          <span className="font-semibold text-slate-200">CineForge ML Engine</span>
          <span className="text-slate-400">• MovieLens ml-latest-small</span>
        </div>
        <div className="flex gap-6 text-slate-400">
          <span>Content-Based (TF-IDF)</span>
          <span>Item-Based Collaborative Filtering</span>
          <span>PostgreSQL Integrated</span>
        </div>
      </div>
    </footer>
  );
};
