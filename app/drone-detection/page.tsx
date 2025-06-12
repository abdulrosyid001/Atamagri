"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AlertCircle, Battery, Camera, Compass, Loader2, Play, Square, Map, Settings, Wifi, WifiOff } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import MissionPlanner from "./mission-planner"

const API_BASE = process.env.NEXT_PUBLIC_DRONE_API_URL || "http://localhost:8000"

interface DroneStatus {
  battery: number
  altitude: number
  speed: number
  signal: number
  isConnected: boolean
}

export default function DroneDetectionPage() {
  const videoRef = useRef<HTMLImageElement>(null)
  const [running, setRunning] = useState(false)
  const [prediction, setPrediction] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [source, setSource] = useState<'tello' | 'webcam'>('tello')
  const [droneStatus, setDroneStatus] = useState<DroneStatus>({
    battery: 100,
    altitude: 0,
    speed: 0,
    signal: 100,
    isConnected: false
  })
  const [activeTab, setActiveTab] = useState("live")

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

  // Poll drone status every second
  useEffect(() => {
    let interval: NodeJS.Timeout
    if (running && source === 'tello') {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/drone_status`)
          const data = await res.json()
          setDroneStatus(data)
        } catch (err) {
          console.error(err)
          setError("Failed to fetch drone status")
        }
      }, 1000)
    }
    return () => clearInterval(interval)
  }, [running, source])

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

  const handleDroneCommand = async (command: string) => {
    try {
      await fetch(`${API_BASE}/drone_command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command })
      })
    } catch (err) {
      setError("Failed to send drone command")
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-50 p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6 text-green-700">Drone Control Center</h1>

        {error && (
          <div className="flex items-center text-red-600 mb-4 bg-red-50 p-3 rounded-lg">
            <AlertCircle className="w-5 h-5 mr-2" /> {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Video Feed */}
          <div className="lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle>Live Feed</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="w-full aspect-video bg-gray-200 rounded-lg flex items-center justify-center overflow-hidden">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img ref={videoRef} alt="Live frame" className="object-contain w-full h-full" />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Drone Status */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Drone Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Battery className="w-5 h-5 mr-2 text-gray-600" />
                    <span>Battery</span>
                  </div>
                  <Progress value={droneStatus.battery} className="w-32" />
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Compass className="w-5 h-5 mr-2 text-gray-600" />
                    <span>Altitude</span>
                  </div>
                  <span>{droneStatus.altitude}m</span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Wifi className="w-5 h-5 mr-2 text-gray-600" />
                    <span>Signal</span>
                  </div>
                  <Progress value={droneStatus.signal} className="w-32" />
                </div>
              </CardContent>
            </Card>

            {/* Controls */}
            <Card>
              <CardHeader>
                <CardTitle>Controls</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-2">
                  <Button variant="outline" onClick={() => handleDroneCommand('takeoff')}>Take Off</Button>
                  <Button variant="outline" onClick={() => handleDroneCommand('land')}>Land</Button>
                  <Button variant="outline" onClick={() => handleDroneCommand('emergency')}>Emergency</Button>
                  <Button variant="outline" onClick={() => handleDroneCommand('up')}>Up</Button>
                  <Button variant="outline" onClick={() => handleDroneCommand('down')}>Down</Button>
                  <Button variant="outline" onClick={() => handleDroneCommand('flip')}>Flip</Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Tabs for different features */}
        <div className="mt-6">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList>
              <TabsTrigger value="live">Live Detection</TabsTrigger>
              <TabsTrigger value="mission">Mission Planning</TabsTrigger>
              <TabsTrigger value="settings">Settings</TabsTrigger>
            </TabsList>
            <TabsContent value="live" className="mt-4">
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center space-x-4">
                    <select 
                      value={source} 
                      onChange={e => setSource(e.target.value as any)} 
                      className="border rounded px-3 py-2"
                    >
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
                    {prediction && (
                      <div className="flex items-center space-x-2">
                        <span className="font-medium">Prediction:</span>
                        <span className="text-green-600">{prediction}</span>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="mission" className="mt-4">
              <Card>
                <CardContent className="pt-6">
                  <MissionPlanner />
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="settings" className="mt-4">
              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span>Detection Sensitivity</span>
                      <input type="range" min="0" max="100" className="w-32" />
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Camera Resolution</span>
                      <select className="border rounded px-2 py-1">
                        <option>720p</option>
                        <option>1080p</option>
                      </select>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  )
}
