import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { UserProvider } from "@/context/UserContext";
import { ToastProvider } from "@/context/ToastContext";

export const metadata: Metadata = {
  title: "CineForge ML — Hybrid Movie Recommendation System",
  description: "Production-style movie recommendation engine combining Content-Based and Collaborative Filtering models.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-100 min-h-screen flex flex-col antialiased selection:bg-amber-500/30 selection:text-amber-200">
        <UserProvider>
          <ToastProvider>
            <Navbar />
            <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
              {children}
            </main>
            <Footer />
          </ToastProvider>
        </UserProvider>
      </body>
    </html>
  );
}
