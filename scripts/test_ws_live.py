# scripts/test_ws_live.py
import asyncio
import websockets

async def main():
    session_id = input("paste session_id: ")
    async with websockets.connect(f"ws://localhost:8000/ws/{session_id}") as ws:
        print("connected, waiting for live events...")
        while True:
            msg = await ws.recv()
            print("live:", msg)

asyncio.run(main())