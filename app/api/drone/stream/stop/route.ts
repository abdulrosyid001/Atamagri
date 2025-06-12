import { NextResponse } from "next/server"
import { Tello } from "djitellopy"

let tello: Tello | null = null
let wss: WebSocketServer | null = null
let streamInterval: NodeJS.Timeout | null = null

export async function GET() {
  try {
    if (streamInterval) {
      clearInterval(streamInterval)
      streamInterval = null
    }

    if (wss) {
      wss.close()
      wss = null
    }

    if (tello) {
      await tello.streamoff()
    }

    return NextResponse.json({
      success: true,
      message: "Stream stopped"
    })
  } catch (error) {
    console.error("Failed to stop stream:", error)
    return NextResponse.json({
      success: false,
      message: "Failed to stop stream"
    }, { status: 500 })
  }
} 