import type { Metadata } from "next";
import Link from "next/link";
import { TrainFront } from "lucide-react";
import "leaflet/dist/leaflet.css";
import "./globals.css";
import "./routes.css";

export const metadata: Metadata = {
  title: "RaahAI | Train route finder",
  description: "Find an Indian train by name or number and explore its recorded route and station timetable.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  // Dark Reader can add attributes to the root element before React hydrates.
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <header className="atlas-header"><div className="atlas-wrap masthead">
          <Link href="/" className="atlas-brand"><TrainFront size={23} aria-hidden="true" /> RaahAI</Link>
          <span>Indian Railways route finder</span>
        </div></header>
        <main>{children}</main>
      </body>
    </html>
  );
}
