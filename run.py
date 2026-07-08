import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("APP_HOST", "127.0.0.1")
PORT = int(os.getenv("APP_PORT", 8000))
RELOAD = os.getenv("APP_RELOAD", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        log_level=LOG_LEVEL
    )
    