import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Vazirmatn } from "next/font/google";
import "./globals.css";

// 1. Initialize Vazirmatn as a CSS variable
const vazir = Vazirmatn({
  subsets: ["arabic", "latin"],
  variable: "--font-vazir", // Define the variable name
  display: 'swap', // Recommended for font performance
});

// 2. Initialize Geist Sans as a CSS variable
const geistSans = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans", // Define the variable name
  display: 'swap',
});

// 3. Initialize Geist Mono as a CSS variable
const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono", // Define the variable name
  display: 'swap',
});

export const metadata: Metadata = {
  title: "Attention Planner - AI-Powered Task Management",
  description: "An advanced todo planner that leverages neural networks to help you focus on what matters most. Optimize your productivity with intelligent task prioritization.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fa" dir="rtl">
      {/* 3. Apply font variables to the root element */}
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${vazir.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
