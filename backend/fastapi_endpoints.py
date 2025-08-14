from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

APP_HOST = config["LOCAL"]["APP_HOST"]
APP_PORT = config["LOCAL"]["APP_PORT"]

app = FastAPI()
# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/save-self-description")
async def save_self_description(user_id: str,description: str):
    pass

@app.post("/save-partner-description")
async def save_partner_description(user_id: str,description: str):
    pass

@app.post("/best-matches")
async def get_best_match(user_id:str, top_n: int):
    pass

if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level="info")
