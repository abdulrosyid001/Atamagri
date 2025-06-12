"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { MapPin, Plus, Trash2 } from "lucide-react"

interface Waypoint {
  id: number
  latitude: number
  longitude: number
  altitude: number
}

export default function MissionPlanner() {
  const [waypoints, setWaypoints] = useState<Waypoint[]>([])
  const [newWaypoint, setNewWaypoint] = useState({
    latitude: 0,
    longitude: 0,
    altitude: 10
  })

  const addWaypoint = () => {
    setWaypoints([
      ...waypoints,
      {
        id: Date.now(),
        ...newWaypoint
      }
    ])
    setNewWaypoint({
      latitude: 0,
      longitude: 0,
      altitude: 10
    })
  }

  const removeWaypoint = (id: number) => {
    setWaypoints(waypoints.filter(wp => wp.id !== id))
  }

  const saveMission = async () => {
    try {
      const response = await fetch("/api/missions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ waypoints })
      })
      if (!response.ok) throw new Error("Failed to save mission")
      // Handle success
    } catch (error) {
      // Handle error
      console.error(error)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Add Waypoint</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label htmlFor="latitude">Latitude</Label>
              <Input
                id="latitude"
                type="number"
                value={newWaypoint.latitude}
                onChange={e => setNewWaypoint({ ...newWaypoint, latitude: parseFloat(e.target.value) })}
                step="0.000001"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="longitude">Longitude</Label>
              <Input
                id="longitude"
                type="number"
                value={newWaypoint.longitude}
                onChange={e => setNewWaypoint({ ...newWaypoint, longitude: parseFloat(e.target.value) })}
                step="0.000001"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="altitude">Altitude (m)</Label>
              <Input
                id="altitude"
                type="number"
                value={newWaypoint.altitude}
                onChange={e => setNewWaypoint({ ...newWaypoint, altitude: parseFloat(e.target.value) })}
                min="0"
                max="100"
              />
            </div>
          </div>
          <Button onClick={addWaypoint} className="mt-4">
            <Plus className="w-4 h-4 mr-2" /> Add Waypoint
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Mission Waypoints</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {waypoints.map((waypoint, index) => (
              <div key={waypoint.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-4">
                  <MapPin className="w-5 h-5 text-gray-500" />
                  <div>
                    <div className="font-medium">Waypoint {index + 1}</div>
                    <div className="text-sm text-gray-500">
                      {waypoint.latitude.toFixed(6)}, {waypoint.longitude.toFixed(6)} - {waypoint.altitude}m
                    </div>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => removeWaypoint(waypoint.id)}
                  className="text-red-500 hover:text-red-700"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </div>
            ))}
            {waypoints.length === 0 && (
              <div className="text-center text-gray-500 py-8">
                No waypoints added yet. Add waypoints to create a mission.
              </div>
            )}
          </div>
          {waypoints.length > 0 && (
            <Button onClick={saveMission} className="mt-6 w-full">
              Save Mission
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  )
} 