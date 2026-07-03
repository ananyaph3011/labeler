import os
import zipfile
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI()

# Enable CORS globally so any client/browser can interface with the API securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DYNAMIC PATH OPTIMIZATION: Resolves the folder location relative to where this script lives.
# This eliminates hardcoded "C:\Users\..." paths so it works out-of-the-box on cloud environments.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIST = os.path.join(BASE_DIR, "dist")

@app.post("/api/export-zip")
async def export_zip(
    zip_name: str = Form(...),
    images: List[UploadFile] = File(...),
    annotations: List[str] = Form(...),
    filenames: List[str] = Form(...)
):
    memory_file = BytesIO()
    
    # Compress files with ZIP_DEFLATED to maximize speed and minimize download package size
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. Package images into an internal "images/" directory
        for img in images:
            contents = await img.read()
            zf.writestr(f"images/{img.filename}", contents)
            
        # 2. Package corresponding label configurations into "annotations/"
        for anno_content, fname in zip(annotations, filenames):
            zf.writestr(f"annotations/{fname}", anno_content)
                
    # PROGRESS BAR OPTIMIZATION: Check the stream size in bytes before pushing to the client.
    # Passing this header allows the browser to calculate true download percentages.
    file_size = memory_file.tell()
    memory_file.seek(0)
    
    return StreamingResponse(
        memory_file, 
        media_type="application/x-zip-compressed", 
        headers={
            "Content-Disposition": f"attachment; filename={zip_name}.zip",
            "Content-Length": str(file_size)
        }
    )

# Serve the static production application files if compiled frontend assets exist
if os.path.exists(FRONTEND_DIST):
    if os.path.exists(os.path.join(FRONTEND_DIST, "assets")):
        app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))

if __name__ == "__main__":
    import uvicorn
    # Local fallback execution handler. Listens on 0.0.0.0 so teammates on your network can connect.
    print(f"Initializing local runtime server. Deployment target: {FRONTEND_DIST}")
    uvicorn.run(app, host="0.0.0.0", port=8000)