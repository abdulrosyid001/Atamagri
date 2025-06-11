"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import Image from "next/image"
import { AlertCircle, Loader2, Play, Square } from "lucide-react"

const API_BASE = process.env.NEXT_PUBLIC_DRONE_API_URL || "http://localhost:8000"

export default function DroneDetectionPage() {
  const videoRef = useRef<HTMLImageElement>(null)
  const [running, setRunning] = useState(false)
  const [prediction, setPrediction] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [source, setSource] = useState<'tello' | 'webcam'>('tello')

  // Poll latest frame every 300 ms when running
  useEffect(() => {
    let interval: NodeJS.Timeout
    if (running) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/latest_frame`)
          if (res.status === 204) return // no frame yet
          const blob = await res.blob()
          const url = URL.createObjectURL(blob)
          if (videoRef.current) videoRef.current.src = url
        } catch (err) {
          console.error(err)
          setError("Failed to fetch frame")
        }
      }, 300)
    }
    return () => clearInterval(interval)
  }, [running])

  // Poll prediction every second
  useEffect(() => {
    let interval: NodeJS.Timeout
    if (running) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/latest_prediction`)
          const data = await res.json()
          setPrediction(data.prediction)
        } catch (err) {
          console.error(err)
          setError("Failed to fetch prediction")
        }
      }, 1000)
    }
    return () => clearInterval(interval)
  }, [running])

  const start = async () => {
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/start?source=${source}`)
      if (res.ok) setRunning(true)
      else setError("Unable to start detector")
    } catch {
      setError("API unreachable")
    }
  }

  const stop = async () => {
    setError(null)
    try {
      await fetch(`${API_BASE}/stop`)
    } catch {
      /* ignore */
    } finally {
      setRunning(false)
      setPrediction(null)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-50 flex flex-col items-center py-10">
      <h1 className="text-3xl font-bold mb-6 text-green-700">Drone Plant Disease Detection</h1>

      {error && (
        <div className="flex items-center text-red-600 mb-4">
          <AlertCircle className="w-5 h-5 mr-2" /> {error}
        </div>
      )}

      <div className="w-[640px] h-[480px] bg-gray-200 rounded shadow flex items-center justify-center overflow-hidden">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img ref={videoRef} alt="Live frame" className="object-contain" />
      </div>

      <div className="mt-4 flex items-center space-x-4">
        <select value={source} onChange={e => setSource(e.target.value as any)} className="border rounded px-2 py-1">
          <option value="tello">Tello Drone</option>
          <option value="webcam">Webcam</option>
        </select>
        {!running ? (
          <Button onClick={start} className="bg-green-600 hover:bg-green-700">
            <Play className="w-4 h-4 mr-2" /> Start Detection
          </Button>
        ) : (
          <Button onClick={stop} variant="destructive">
            <Square className="w-4 h-4 mr-2" /> Stop
          </Button>
        )}
        {running && !prediction && <Loader2 className="animate-spin text-gray-600" />}
        {prediction && <span className="font-medium text-gray-700">Prediction: {prediction}</span>}
      </div>

      <p className="text-sm text-gray-500 mt-8 max-w-xl text-center">
        This demo streams live video from a connected webcam (or Tello drone if configured) to the backend, runs a
        PyTorch model to classify plant diseases, and returns the latest frame and prediction to the browser every few
        seconds.
      </p>
    </div>
  )
}
