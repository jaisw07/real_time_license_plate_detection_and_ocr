from fastapi import APIRouter, Request, Query
from fastapi.staticfiles import StaticFiles
import os

router = APIRouter()

# Serve static images from history
images_path = "data/history/images"
os.makedirs(images_path, exist_ok=True)

@router.get("/")
async def get_history(
    request: Request,
    limit: int = Query(50, ge=1, le=100)
):
    history_manager = request.app.state.history_manager
    history = history_manager.get_history(limit=limit)
    return history
