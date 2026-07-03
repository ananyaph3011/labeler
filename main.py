import os
import json
import zipfile
import uuid
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIST = os.path.join(BASE_DIR, "dist")

STORAGE_DIR = os.path.join(BASE_DIR, "storage")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports") # New folder for compiled ZIPs
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

# Dictionary to track background export progress
export_jobs = {}

# ... (Keep your existing @app.post("/api/save-stage") endpoint exactly the same) ...

# ---------------------------------------------------------
# BACKGROUND TASK: Compresses the dataset on the disk, not RAM
# ---------------------------------------------------------
def compile_dataset_task(job_id: str, safe_name: str, export_format: str):
    try:
        dataset_dir = os.path.join(STORAGE_DIR, safe_name)
        images_dir = os.path.join(dataset_dir, "images")
        labels_dir = os.path.join(dataset_dir, "labels")
        
        # Save directly to disk to save memory and speed up processing
        zip_filepath = os.path.join(EXPORTS_DIR, f"{job_id}.zip")
        
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
            image_files = os.listdir(images_dir) if os.path.exists(images_dir) else []
            total_files = len(image_files)
            
            # Write images
            for i, fname in enumerate(image_files):
                zf.write(os.path.join(images_dir, fname), f"images/{fname}")
                # Update progress tracker
                export_jobs[job_id]["progress"] = int((i / total_files) * 50) 
                
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

            # YOLO Formatting
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

            # COCO Formatting
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


# 1. Trigger the export (Fast, no timeout)
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
    
    # Hand off the heavy lifting to the background thread
    background_tasks.add_task(compile_dataset_task, job_id, safe_name, format)
    
    return {"job_id": job_id}


# 2. Check the progress
@app.get("/api/export-status/{job_id}")
async def check_export_status(job_id: str):
    if job_id not in export_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return export_jobs[job_id]


# 3. Download the final compiled file from disk
@app.get("/api/download/{job_id}/{filename}")
async def download_export(job_id: str, filename: str):
    file_path = os.path.join(EXPORTS_DIR, f"{job_id}.zip")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # FileResponse handles massive files efficiently without buffering into RAM
    return FileResponse(file_path, media_type="application/x-zip-compressed", filename=filename)


# ... (Keep your existing static file serving setup at the bottom) ...