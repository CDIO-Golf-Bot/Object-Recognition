import asyncio
import websockets

async def send_command(command):
    uri = "ws://10.135.97.57:8765"  # IP address of your EV3 robot
    async with websockets.connect(uri) as websocket:
        await websocket.send(command)
        response = await websocket.recv()
        print(f"Response: {response}")

# Send commands dynamically
asyncio.run(send_command("move_forward"))
asyncio.run(send_command("turn_left"))
