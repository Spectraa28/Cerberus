import asyncio
from cerberus.runtime.session import EventLog

async def main():
    log = EventLog()  # same default db_path as the gateway
    await log.connect()
    session_id = await log.create_session(label="live broadcast test")
    print("session_id:", session_id)
    await log.close()

asyncio.run(main())