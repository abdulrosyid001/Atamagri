import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import Link from "next/link"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "Atamagri - Climate Intelligence for Smart Farming",
  description:
    "Real-time weather monitoring, climate data analytics, and AI-driven decision recommendations for farmers, fishermen, and researchers with AtamaStation IoT platform.",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <nav>
          {/* Drone link only visible inside dashboard to simplify nav */}
        </nav>
        {children}
      </body>
    </html>
  )
}
