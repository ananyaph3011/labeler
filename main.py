import os
import json
import zipfile
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
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

# Permanent server directory to temporarily cache incoming files during labeling
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

@app.post("/api/save-stage")
async def save_stage(
    zip_name: str = Form(...),
    filename: str = Form(...),
    boxes: str = Form(...),  # Received as a stringified JSON array from frontend
    image: UploadFile = File(...)
):
    try:
        # Sanitize workspace name to prevent directory traversal attacks
        safe_name = "".join([c for c in zip_name if c.isalnum() or c in ('_', '-')]).strip()
        if not safe_name:
            safe_name = "default_dataset"

        # Create localized staging paths
        dataset_dir = os.path.join(STORAGE_DIR, safe_name)
        images_dir = os.path.join(dataset_dir, "images")
        labels_dir = os.path.join(dataset_dir, "labels")
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # 1. Stream and write the single image directly to server disk storage
        image_path = os.path.join(images_dir, filename)
        contents = await image.read()
        with open(image_path, "wb") as f:
            f.write(contents)

        # 2. Write raw bounding box coordinates to a matching label json file
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(labels_dir, f"{base_name}.json")
        with open(label_path, "w") as f:
            f.write(boxes)

        return {"status": "success", "message": f"Successfully cached {filename} on server."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export-zip")
async def export_zip(
    zip_name: str = Form(...),
    format: str = Form(...)  # Expects 'yolo' or 'coco'
):
    safe_name = "".join([c for c in zip_name if c.isalnum() or c in ('_', '-')]).strip()
    dataset_dir = os.path.join(STORAGE_DIR, safe_name)

    if not os.path.exists(dataset_dir):
        raise HTTPException(status_code=404, detail="Workspace archive empty. Please save at least one annotation.")

    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    memory_file = BytesIO()

    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Append all uploaded images found inside server memory cache
        image_files = os.listdir(images_dir) if os.path.exists(images_dir) else []
        for fname in image_files:
            zf.write(os.path.join(images_dir, fname), f"images/{fname}")

        # Scan all raw JSON annotations to determine categories dynamically
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

        # Compile annotations based on selected format
        if format == "yolo":
            for l_file, boxes_data in parsed_labels.items():
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
            
            # Export helper label file
            zf.writestr("classes.txt", "\n".join(class_list))

        else:  # COCO Format compilation logic
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
                        "id": anno_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [b["x"], b["y"], b["w"], b["h"]],
                        "area": b["w"] * b["h"],
                        "iscrowd": 0,
                        "segmentation": []
                    })
                    anno_id += 1
            
            zf.writestr("annotations.json", json.dumps(coco, indent=2))

    file_size = memory_file.tell()
    memory_file.seek(0)

    return StreamingResponse(
        memory_file, 
        media_type="application/x-zip-compressed", 
        headers={
            "Content-Disposition": f"attachment; filename={safe_name}.zip",
            "Content-Length": str(file_size)
        }
    )

if os.path.exists(FRONTEND_DIST):
    if os.path.exists(os.path.join(FRONTEND_DIST, "assets")):
        app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))