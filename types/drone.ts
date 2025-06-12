export interface DroneStatus {
  connected: boolean
  mode: string
  battery: number
  signal: number
}

export interface Prediction {
  class: string
  confidence: number
} 