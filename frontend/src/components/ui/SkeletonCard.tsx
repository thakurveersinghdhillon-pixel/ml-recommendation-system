import React from "react";

export const SkeletonCard: React.FC = () => {
  return (
    <div className="bg-slate-900/60 rounded-2xl border border-slate-800 p-4 flex flex-col gap-3 animate-pulse">
      <div className="h-44 w-full bg-slate-800/80 rounded-xl" />
      <div className="h-4 w-3/4 bg-slate-800/80 rounded" />
      <div className="h-3 w-1/2 bg-slate-800/60 rounded" />
      <div className="flex gap-2 pt-2">
        <div className="h-6 w-16 bg-slate-800/80 rounded-md" />
        <div className="h-6 w-16 bg-slate-800/80 rounded-md" />
      </div>
    </div>
  );
};
