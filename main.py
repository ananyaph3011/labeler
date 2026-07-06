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

BASE_DIR = str(Path(__file__).resolve().parent)
FRONTEND_DIST = os.path.join(BASE_DIR, "dist")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

export_jobs = {}

@app.post("/api/save-stage")
async def save_stage(
    session_id: str = Form(...),
    zip_name: str = Form(...),
    filename: str = Form(...),
    boxes: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        # FIX: Isolate directories by session_id, NOT zip_name, to prevent data mixing
        dataset_dir = os.path.join(STORAGE_DIR, session_id)
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

def compile_dataset_task(job_id: str, session_id: str, safe_zip_name: str, export_format: str):
    try:
        dataset_dir = os.path.join(STORAGE_DIR, session_id)
        images_dir = os.path.join(dataset_dir, "images")
        labels_dir = os.path.join(dataset_dir, "labels")
        
        zip_filepath = os.path.join(EXPORTS_DIR, f"{job_id}.zip")
        
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
            
            # FIX: We now base everything STRICTLY on the images present. 
            image_files = os.listdir(images_dir) if os.path.exists(images_dir) else []
            total_files = len(image_files)
            
            # 1. First Pass: Read all JSONs to gather a universal Class List
            parsed_labels = {}
            all_classes = set()
            if os.path.exists(labels_dir):
                for l_file in os.listdir(labels_dir):
                    with open(os.path.join(labels_dir, l_file), "r") as f:
                        data = json.load(f)
                        parsed_labels[l_file] = data
                        for b in data:
                            if "isEmpty" not in b: # ignore empty dummy boxes from sync
                                all_classes.add(b["className"])
            
            class_list = sorted(list(all_classes))

            # 2. Second Pass: Write out strictly 1-to-1 matched files
            if export_format == "yolo":
                for i, fname in enumerate(image_files):
                    base_name = os.path.splitext(fname)[0]
                    zf.write(os.path.join(images_dir, fname), f"{safe_zip_name}/images/{fname}")
                    
                    yolo_lines = []
                    boxes_data = parsed_labels.get(f"{base_name}.json", [])
                    
                    for b in boxes_data:
                        if "isEmpty" not in b:
                            cat_id = class_list.index(b["className"])
                            x_center = (b["x"] + b["w"] / 2) / b["imgW"]
                            y_center = (b["y"] + b["h"] / 2) / b["imgH"]
                            norm_w = b["w"] / b["imgW"]
                            norm_h = b["h"] / b["imgH"]
                            yolo_lines.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                    
                    # This guarantees an empty .txt file is written even if there are no annotations
                    zf.writestr(f"{safe_zip_name}/labels/{base_name}.txt", "\n".join(yolo_lines))
                    export_jobs[job_id]["progress"] = int((i / total_files) * 90)
                
                zf.writestr(f"{safe_zip_name}/classes.txt", "\n".join(class_list))

            else:  
                # COCO Format 1-to-1 strict mapping
                coco = {"images": [], "annotations": [], "categories": []}
                for i, cls_name in enumerate(class_list):
                    coco["categories"].append({"id": i, "name": cls_name, "supercategory": "none"})

                anno_id = 1
                for img_id, fname in enumerate(image_files):
                    base_name = os.path.splitext(fname)[0]
                    zf.write(os.path.join(images_dir, fname), f"{safe_zip_name}/images/{fname}")
                    
                    boxes_data = parsed_labels.get(f"{base_name}.json", [])
                    
                    # Extract dimensions (from dummy payload or real boxes)
                    img_w, img_h = 800, 600 # default fallback
                    if boxes_data and len(boxes_data) > 0:
                        img_w = boxes_data[0].get("imgW", 800)
                        img_h = boxes_data[0].get("imgH", 600)

                    coco["images"].append({"id": img_id, "file_name": fname, "width": img_w, "height": img_h})

                    for b in boxes_data:
                        if "isEmpty" not in b:
                            cat_id = class_list.index(b["className"])
                            coco["annotations"].append({
                                "id": anno_id, "image_id": img_id, "category_id": cat_id,
                                "bbox": [b["x"], b["y"], b["w"], b["h"]], "area": b["w"] * b["h"],
                                "iscrowd": 0, "segmentation": []
                            })
                            anno_id += 1
                            
                    export_jobs[job_id]["progress"] = int((img_id / total_files) * 90)
                
                zf.writestr(f"{safe_zip_name}/annotations.json", json.dumps(coco, indent=2))

        export_jobs[job_id]["progress"] = 100
        export_jobs[job_id]["status"] = "completed"
        export_jobs[job_id]["download_url"] = f"/api/download/{job_id}/{safe_zip_name}.zip"

    except Exception as e:
        export_jobs[job_id]["status"] = "failed"
        export_jobs[job_id]["error"] = str(e)


@app.post("/api/request-export")
async def request_export(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    zip_name: str = Form(...),
    format: str = Form(...)
):
    safe_name = "".join([c for c in zip_name if c.isalnum() or c in ('_', '-')]).strip()
    if not safe_name: safe_name = "dataset"

    job_id = str(uuid.uuid4())
    export_jobs[job_id] = {"status": "processing", "progress": 0}
    
    background_tasks.add_task(compile_dataset_task, job_id, session_id, safe_name, format)
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
# UI MOUNTING
# ---------------------------------------------------------
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="ui")
else:
    @app.get("/")
    def fallback():
        return {"error": f"The compiled frontend was not found at {FRONTEND_DIST}. Ensure 'dist' is uploaded to GitHub."}