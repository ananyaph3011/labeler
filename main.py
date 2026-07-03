import os
import json
import zipfile
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. BULLETPROOF PATH RESOLUTION (Fixes the URI Error)
# Using pathlib guarantees a strict absolute path on any OS.
# ---------------------------------------------------------
BASE_DIR = str(Path(__file__).resolve().parent)

FRONTEND_DIST = os.path.join(BASE_DIR, "dist")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

export_jobs = {}

@app.post("/api/save-stage")
async def save_stage(
    zip_name: str = Form(...),
    filename: str = Form(...),
    boxes: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        safe_name = "".join([c for c in zip_name if c.isalnum() or c in ('_', '-')]).strip()
        if not safe_name: safe_name = "default_dataset"

        dataset_dir = os.path.join(STORAGE_DIR, safe_name)
        images_dir = os.path.join(dataset_dir, "images")
        labels_dir = os.path.join(dataset_dir, "labels")
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        image_path = os.path.join(images_dir, filename)
        contents = await image.read()
        with open(image_path, "wb") as f:
            f.write(contents)

        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(labels_dir, f"{base_name}.json")
        with open(label_path, "w") as f:
            f.write(boxes)

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def compile_dataset_task(job_id: str, safe_name: str, export_format: str):
    try:
        dataset_dir = os.path.join(STORAGE_DIR, safe_name)
        images_dir = os.path.join(dataset_dir, "images")
        labels_dir = os.path.join(dataset_dir, "labels")
        
        zip_filepath = os.path.join(EXPORTS_DIR, f"{job_id}.zip")
        
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
            image_files = os.listdir(images_dir) if os.path.exists(images_dir) else []
            total_files = len(image_files)
            
            for i, fname in enumerate(image_files):
                zf.write(os.path.join(images_dir, fname), f"images/{fname}")
                export_jobs[job_id]["progress"] = int((i / total_files) * 50) if total_files > 0 else 50
                
            label_files = os.listdir(labels_dir) if os.path.exists(labels_dir) else []
            parsed_labels = {}
            all_classes = set()

            for l_file in label_files:
                with open(os.path.join(labels_dir, l_file), "r") as f:
                    data = json.load(f)
                    parsed_labels[l_file] = data
                    for b in data:
                        all_classes.add(b["className"])
            
            class_list = sorted(list(all_classes))

            if export_format == "yolo":
                for i, (l_file, boxes_data) in enumerate(parsed_labels.items()):
                    base_name = os.path.splitext(l_file)[0]
                    yolo_lines = []
                    for b in boxes_data:
                        cat_id = class_list.index(b["className"])
                        x_center = (b["x"] + b["w"] / 2) / b["imgW"]
                        y_center = (b["y"] + b["h"] / 2) / b["imgH"]
                        norm_w = b["w"] / b["imgW"]
                        norm_h = b["h"] / b["imgH"]
                        yolo_lines.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                    
                    zf.writestr(f"annotations/{base_name}.txt", "\n".join(yolo_lines))
                    export_jobs[job_id]["progress"] = 50 + int((i / len(parsed_labels)) * 40)
                
                zf.writestr("classes.txt", "\n".join(class_list))

            else:  
                coco = {"images": [], "annotations": [], "categories": []}
                for i, cls_name in enumerate(class_list):
                    coco["categories"].append({"id": i, "name": cls_name, "supercategory": "none"})

                anno_id = 1
                for img_id, fname in enumerate(image_files):
                    base_name = os.path.splitext(fname)[0]
                    boxes_data = parsed_labels.get(f"{base_name}.json", [])
                    img_w = boxes_data[0]["imgW"] if boxes_data else 800
                    img_h = boxes_data[0]["imgH"] if boxes_data else 600

                    coco["images"].append({"id": img_id, "file_name": fname, "width": img_w, "height": img_h})

                    for b in boxes_data:
                        cat_id = class_list.index(b["className"])
                        coco["annotations"].append({
                            "id": anno_id, "image_id": img_id, "category_id": cat_id,
                            "bbox": [b["x"], b["y"], b["w"], b["h"]], "area": b["w"] * b["h"],
                            "iscrowd": 0, "segmentation": []
                        })
                        anno_id += 1
                
                zf.writestr("annotations.json", json.dumps(coco, indent=2))
                export_jobs[job_id]["progress"] = 90

        export_jobs[job_id]["progress"] = 100
        export_jobs[job_id]["status"] = "completed"
        export_jobs[job_id]["download_url"] = f"/api/download/{job_id}/{safe_name}.zip"

    except Exception as e:
        export_jobs[job_id]["status"] = "failed"
        export_jobs[job_id]["error"] = str(e)


@app.post("/api/request-export")
async def request_export(
    background_tasks: BackgroundTasks,
    zip_name: str = Form(...),
    format: str = Form(...)
):
    safe_name = "".join([c for c in zip_name if c.isalnum() or c in ('_', '-')]).strip()
    dataset_dir = os.path.join(STORAGE_DIR, safe_name)

    if not os.path.exists(dataset_dir):
        raise HTTPException(status_code=404, detail="Workspace archive empty. Save at least one annotation.")

    job_id = str(uuid.uuid4())
    export_jobs[job_id] = {"status": "processing", "progress": 0}
    
    background_tasks.add_task(compile_dataset_task, job_id, safe_name, format)
    return {"job_id": job_id}


@app.get("/api/export-status/{job_id}")
async def check_export_status(job_id: str):
    if job_id not in export_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return export_jobs[job_id]


@app.get("/api/download/{job_id}/{filename}")
async def download_export(job_id: str, filename: str):
    file_path = os.path.join(EXPORTS_DIR, f"{job_id}.zip")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="application/x-zip-compressed", filename=filename)


# ---------------------------------------------------------
# 2. UI MOUNTING (MUST BE AT THE VERY BOTTOM)
# ---------------------------------------------------------
if os.path.exists(FRONTEND_DIST):
    # Forced string casting ensures FastAPI never gets an empty/invalid object
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="ui")
else:
    @app.get("/")
    def fallback():
        return {"error": f"The compiled frontend was not found at {FRONTEND_DIST}. Ensure 'dist' is uploaded to GitHub."}