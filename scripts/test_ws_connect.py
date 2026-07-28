# test_ws_connect.py — temporary
import asyncio
import websockets

async def main():
    session_id = input("paste session_id: ")
    async with websockets.connect(f"ws://localhost:8000/ws/{session_id}") as ws:
        # Should receive the 2 seeded events immediately on connect
        for _ in range(2):
            msg = await ws.recv()
            print("received:", msg)

asyncio.run(main())