import { NextResponse } from "next/server"
import { Tello } from "djitellopy"

let tello: Tello | null = null

const COMMANDS = {
  takeoff: async (tello: Tello) => await tello.takeoff(),
  land: async (tello: Tello) => await tello.land(),
  emergency: async (tello: Tello) => await tello.emergency(),
  up: async (tello: Tello) => await tello.move_up(30),
  down: async (tello: Tello) => await tello.move_down(30),
  left: async (tello: Tello) => await tello.move_left(30),
  right: async (tello: Tello) => await tello.move_right(30),
  forward: async (tello: Tello) => await tello.move_forward(30),
  back: async (tello: Tello) => await tello.move_back(30),
  flip: async (tello: Tello) => await tello.flip("f"),
  rotate_cw: async (tello: Tello) => await tello.rotate_clockwise(30),
  rotate_ccw: async (tello: Tello) => await tello.rotate_counter_clockwise(30),
}

export async function POST(request: Request) {
  try {
    const { command } = await request.json()

    if (!tello) {
      return NextResponse.json({
        success: false,
        message: "Drone not connected"
      }, { status: 400 })
    }

    if (!(command in COMMANDS)) {
      return NextResponse.json({
        success: false,
        message: "Invalid command"
      }, { status: 400 })
    }

    await COMMANDS[command as keyof typeof COMMANDS](tello)

    return NextResponse.json({
      success: true,
      message: `Command ${command} executed successfully`
    })
  } catch (error) {
    console.error("Failed to execute command:", error)
    return NextResponse.json({
      success: false,
      message: "Failed to execute command"
    }, { status: 500 })
  }
} 