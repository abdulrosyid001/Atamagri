"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle, Play, Square, ArrowUp, ArrowDown, ArrowLeft, ArrowRight, RotateCcw, RotateCw } from "lucide-react"

interface DroneStatus {
  connected: boolean
  battery: number
  flying: boolean
  mode: string
  signal: number
}

interface Prediction {
  plant: string
  condition: string
  confidence: number
}

export default function DroneControl() {
  const [status, setStatus] = useState<DroneStatus>({
    connected: false,
    battery: 0,
    flying: false,
    mode: "disconnected",
    signal: 0
  })
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [error, setError] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    // Connect to drone server
    connectToDrone()
    
    // Cleanup on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  const connectToDrone = async () => {
    try {
      const response = await fetch("http://localhost:8000/connect")
      const data = await response.json()
      
      if (data.success) {
        setStatus(prev => ({ 
          ...prev, 
          connected: true,
          mode: data.mode || "unknown",
          battery: data.battery || 0,
          signal: 100
        }))
        startVideoStream()
        setError(null)
      } else {
        setError(data.message || "Failed to connect to device")
      }
    } catch (err) {
      setError("Error connecting to drone server")
    }
  }

  const startVideoStream = () => {
    const ws = new WebSocket("ws://localhost:8000/ws")
    wsRef.current = ws

    ws.onmessage = (event) => {
      if (event.data instanceof Blob) {
        // Handle video frame
        const url = URL.createObjectURL(event.data)
        console.log("Received frame blob, size:", event.data.size)
        
        // Always try image first (works for both webcam and drone)
        if (imageRef.current) {
          // Clean up previous URL to prevent memory leaks
          if (imageRef.current.src && imageRef.current.src.startsWith('blob:')) {
            URL.revokeObjectURL(imageRef.current.src)
          }
          imageRef.current.src = url
          console.log("Updated image element with new frame")
        }
        
        // Fallback to video element if needed
        if (!imageRef.current && videoRef.current) {
          if (videoRef.current.src && videoRef.current.src.startsWith('blob:')) {
            URL.revokeObjectURL(videoRef.current.src)
          }
          videoRef.current.src = url
          videoRef.current.load()
        }
      } else {
        // Handle prediction data
        try {
          const prediction = JSON.parse(event.data)
          setPrediction(prediction)
        } catch (err) {
          console.error("Error parsing prediction:", err)
        }
      }
    }

    ws.onerror = (error) => {
      console.error("WebSocket error:", error)
      setError("WebSocket connection error")
    }

    ws.onclose = () => {
      setStatus(prev => ({ ...prev, connected: false }))
      console.log("WebSocket connection closed")
    }

    ws.onopen = () => {
      console.log("WebSocket connection opened")
    }
  }

  const sendCommand = async (command: string) => {
    try {
      const response = await fetch("http://localhost:8000/command", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ command }),
      })
      
      const data = await response.json()
      if (!data.success) {
        setError(data.message)
      }
    } catch (err) {
      setError("Error sending command")
    }
  }

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Drone Control</span>
            {status.connected && (
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  status.mode === "tello" ? "bg-blue-500" : "bg-green-500"
                } animate-pulse`}></div>
                <span className="text-sm text-gray-600">
                  {status.mode === "tello" ? "Tello Drone" : 
                   status.mode === "webcam" ? "Webcam Mode" : "Unknown"}
                </span>
              </div>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          
          {status.connected && status.mode === "webcam" && (
            <Alert className="mb-4 border-blue-200 bg-blue-50">
              <AlertCircle className="h-4 w-4 text-blue-600" />
              <AlertTitle className="text-blue-800">Webcam Simulation Mode</AlertTitle>
              <AlertDescription className="text-blue-700">
                Tello drone not available. Using webcam for disease detection demo. 
                Flight controls are simulated.
              </AlertDescription>
            </Alert>
          )}
          
          <div className="grid gap-4">
            <div className="flex justify-center gap-2">
              <Button
                variant="outline"
                size="icon"
                onClick={() => sendCommand("takeoff")}
                disabled={!status.connected || status.flying}
              >
                <Play className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={() => sendCommand("land")}
                disabled={!status.connected || !status.flying}
              >
                <Square className="h-4 w-4" />
              </Button>
            </div>
            
            <div className="grid grid-cols-3 gap-2">
              <div />
              <Button
                variant="outline"
                size="icon"
                onClick={() => sendCommand("up")}
                disabled={!status.connected || !status.flying}
              >
                <ArrowUp className="h-4 w-4" />
              </Button>
              <div />
              
              <Button
                variant="outline"
                size="icon"
                onClick={() => sendCommand("left")}
                disabled={!status.connected || !status.flying}
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={() => sendCommand("down")}
                disabled={!status.connected || !status.flying}
              >
                <ArrowDown className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={() => sendCommand("right")}
                disabled={!status.connected || !status.flying}
              >
                <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
            
            <div className="flex justify-center gap-2">
              <Button
                variant="outline"
                size="icon"
                onClick={() => sendCommand("rotate_ccw")}
                disabled={!status.connected || !status.flying}
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={() => sendCommand("rotate_cw")}
                disabled={!status.connected || !status.flying}
              >
                <RotateCw className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Live Feed</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="aspect-video bg-black rounded-lg overflow-hidden flex items-center justify-center">
            {status.mode === "webcam" ? (
              <img
                ref={imageRef}
                className="w-full h-full object-cover"
                alt="Live webcam feed"
              />
            ) : (
              <video
                ref={videoRef}
                className="w-full h-full object-cover"
                autoPlay
                playsInline
                muted
              />
            )}
            {!status.connected && (
              <div className="absolute inset-0 flex items-center justify-center text-white">
                <div className="text-center">
                  <div className="w-16 h-16 border-4 border-gray-600 border-t-white rounded-full animate-spin mx-auto mb-4"></div>
                  <p>Connecting to device...</p>
                </div>
              </div>
            )}
          </div>
          
          {prediction && (
            <div className="mt-4 p-4 bg-muted rounded-lg">
              <h3 className="font-semibold">Disease Detection</h3>
              <p>Plant: {prediction.plant}</p>
              <p>Condition: {prediction.condition}</p>
              <p>Confidence: {prediction.confidence.toFixed(2)}%</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
} 