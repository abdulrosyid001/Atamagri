import { NextResponse } from "next/server"
import { Tello } from "djitellopy"

let tello: Tello | null = null

export async function GET() {
  try {
    if (!tello) {
      tello = new Tello()
      await tello.connect()
      await tello.streamon()
    }

    const battery = await tello.get_battery()
    
    return NextResponse.json({
      success: true,
      battery,
      message: "Connected to Tello drone"
    })
  } catch (error) {
    console.error("Failed to connect to drone:", error)
    return NextResponse.json({
      success: false,
      message: "Failed to connect to drone"
    }, { status: 500 })
  }
} 