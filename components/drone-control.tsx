"use client"

import { useEffect, useRef, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DroneStatus, Prediction } from "@/types/drone"
import { useToast } from "@/components/ui/use-toast"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

export default function DroneControl() {
  const [isConnected, setIsConnected] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [source, setSource] = useState<'tello' | 'webcam'>('webcam')
  const [status, setStatus] = useState<DroneStatus>({
    connected: false,
    mode: "idle",
    battery: 0,
    signal: 0,
  })
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const { toast } = useToast()

  const connectToDevice = async (selectedSource: 'tello' | 'webcam') => {
    try {
      setIsLoading(true)
      setError(null)
      
      // Stop any existing connection
      await fetch('http://localhost:8000/stop').catch(console.error)
      
      // Try to connect to selected source
      const response = await fetch(`http://localhost:8000/start?source=${selectedSource}`)
      if (!response.ok) {
        throw new Error(`Failed to connect to ${selectedSource}`)
      }
      
      setIsConnected(true)
      setSource(selectedSource)
      setStatus((prev: DroneStatus) => ({
        ...prev,
        connected: true,
        mode: selectedSource,
        battery: 100,
        signal: 100
      }))
    } catch (err) {
      console.error(`Connection error:`, err)
      setError(`Failed to connect to ${selectedSource}`)
      setIsConnected(false)
      setStatus((prev: DroneStatus) => ({
        ...prev,
        connected: false,
        mode: "error"
      }))
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    // Initial connection
    connectToDevice(source)

    return () => {
      // Cleanup on unmount
      fetch('http://localhost:8000/stop').catch(console.error)
    }
  }, [source])

  useEffect(() => {
    if (!isConnected) return

    // Connect to WebSocket for real-time updates
    const ws = new WebSocket('ws://localhost:8000/ws')
    wsRef.current = ws

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'status') {
        setStatus(data.status)
      } else if (data.type === 'prediction') {
        setPrediction(data.prediction)
        if (data.prediction) {
          toast({
            title: "New Prediction",
            description: `${data.prediction.class} (${(data.prediction.confidence * 100).toFixed(1)}% confidence)`,
          })
        }
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setError('Lost connection to server')
    }

    ws.onclose = () => {
      console.log('WebSocket connection closed')
      setError('Connection to server closed')
    }

    return () => {
      ws.close()
    }
  }, [isConnected, toast])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Connecting to {source === 'tello' ? 'Tello drone' : 'webcam'}...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-red-500 mb-4">
            <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <p className="text-red-500 font-semibold">{error}</p>
          <p className="text-gray-600 mt-2">Please check your connection and try again</p>
          <div className="mt-4">
            <Select
              value={source}
              onValueChange={(value: 'tello' | 'webcam') => {
                setSource(value)
                connectToDevice(value)
              }}
            >
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Select source" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="tello">Tello Drone</SelectItem>
                <SelectItem value="webcam">Webcam</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="relative h-full">
      <div className="absolute top-4 right-4 flex items-center space-x-4">
        <Select
          value={source}
          onValueChange={(value: 'tello' | 'webcam') => {
            setSource(value)
            connectToDevice(value)
          }}
        >
          <SelectTrigger className="w-[180px] bg-black bg-opacity-50 text-white border-none">
            <SelectValue placeholder="Select source" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="tello">Tello Drone</SelectItem>
            <SelectItem value="webcam">Webcam</SelectItem>
          </SelectContent>
        </Select>
        <div className="bg-black bg-opacity-50 text-white px-3 py-1 rounded-full text-sm">
          {source === 'tello' ? 'Connected to Tello' : 'Using Webcam'}
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Live Feed</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
              {isLoading ? (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500"></div>
                </div>
              ) : (
                <img
                  src="http://localhost:8000/video_feed"
                  alt="Drone/Webcam Feed"
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    console.error("Error loading video feed");
                    setError("Failed to load video feed. Please check your connection.");
                  }}
                />
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-gray-100 rounded-lg">
                  <p className="text-sm text-gray-500">Mode</p>
                  <p className="text-lg font-semibold">{status.mode}</p>
                </div>
                <div className="p-4 bg-gray-100 rounded-lg">
                  <p className="text-sm text-gray-500">Battery</p>
                  <p className="text-lg font-semibold">{status.battery}%</p>
                </div>
              </div>
              
              {prediction && (
                <div className="p-4 bg-green-50 rounded-lg">
                  <p className="text-sm text-green-500">Latest Prediction</p>
                  <p className="text-lg font-semibold text-green-700">
                    {prediction.class} ({(prediction.confidence * 100).toFixed(1)}%)
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
} 