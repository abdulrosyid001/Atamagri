import { NextResponse } from "next/server"
import { Tello } from "djitellopy"
import { WebSocketServer } from "ws"
import { createServer } from "http"

let tello: Tello | null = null
let wss: WebSocketServer | null = null
let streamInterval: NodeJS.Timeout | null = null

export async function GET() {
  try {
    if (!tello) {
      tello = new Tello()
      await tello.connect()
    }

    if (!wss) {
      const server = createServer()
      wss = new WebSocketServer({ server })
      
      wss.on("connection", (ws) => {
        console.log("Client connected to stream")
        
        // Start sending frames
        streamInterval = setInterval(async () => {
          try {
            const frame = await tello?.get_frame_read().frame
            if (frame) {
              // Convert frame to JPEG
              const buffer = Buffer.from(frame)
              ws.send(buffer)
            }
          } catch (error) {
            console.error("Error sending frame:", error)
          }
        }, 100) // 10 FPS
      })
      
      server.listen(8000)
    }

    return NextResponse.json({
      success: true,
      message: "Stream started"
    })
  } catch (error) {
    console.error("Failed to start stream:", error)
    return NextResponse.json({
      success: false,
      message: "Failed to start stream"
    }, { status: 500 })
  }
} 