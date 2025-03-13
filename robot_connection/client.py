import asyncio
import websockets

async def send_coordinates():
    uri = "ws://<EV3_IP>:8765"  # Replace <EV3_IP> with the EV3 brick's IP address
    async with websockets.connect(uri) as websocket:
        while True:
            coordinates = "100,200"  # Example coordinates (x, y)
            await websocket.send(coordinates)
            print(f"Sent coordinates: {coordinates}")
            await asyncio.sleep(1)  # Send coordinates every second

# Run the client
asyncio.get_event_loop().run_until_complete(send_coordinates())