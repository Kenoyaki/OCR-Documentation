from insightface.app import FaceAnalysis
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import argparse
import pickle
import sys
import cv2

# --- helpers shared with the notebook ---
def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v) + eps
    return v / n

def build_face_gallery(folder, fa, ext=(".jpg", ".png", ".jpeg")):
    folder = Path(folder)
    gallery = {}
    for person_dir in folder.iterdir():
        if not person_dir.is_dir():
            continue
        embs = []
        for img_path in person_dir.glob("*"):
            if img_path.suffix.lower() not in ext:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = fa.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if faces and getattr(faces[0], "embedding", None) is not None:
                embs.append(normalize(faces[0].embedding))
            else:
                # --- NEW CODE: Warning ---
                print(f"Warning: No face detected in {img_path.name}. Image might be too tight or blurry.")
                # -------------------------
        if embs:
            gallery[person_dir.name] = normalize(np.mean(embs, axis=0))
    return gallery

def normalize_plate_text(s: str):
    return "".join(ch for ch in (s or "").upper() if ch.isalnum())

def build_plate_db_from_list(texts):
    return [normalize_plate_text(t) for t in texts if t]

def export_yolo_to_onnx(weights, out_onnx, imgsz=640, device="cpu", half=True):
    model = YOLO(weights)
    kwargs = dict(format="onnx", imgsz=imgsz, device=device, simplify=True, name=out_onnx)
    if half:
        kwargs["half"] = True
    model.export(**kwargs)
    print(f"Saved ONNX to {out_onnx}")

def main():
    parser = argparse.ArgumentParser("Prepare ONNX + galleries")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--onnx", default="face_plate_fp16.onnx")
    parser.add_argument("--faces_dir", default="faces")
    parser.add_argument("--face_db", default="faces.pkl")
    parser.add_argument("--plates_txt", nargs="*", default=[])
    parser.add_argument("--plates_db", default="plates.pkl")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")

    default_cli = [
        "--weights", r"D:\Kenny\IVS\Face and plate recog\runs\detect\face_plate_v2\weights\best.pt",
        "--onnx", r"D:\Kenny\IVS\Face and plate recog\face_plate_fp16.onnx",
        "--faces_dir", "faces",
        "--face_db", "faces.pkl",
        "--plates_txt", "F6797OB", "B2156TOR",
        "--plates_db", "plates.pkl",
        "--imgsz", "640",
        "--device", "cuda:0",
    ]
    args = parser.parse_args(default_cli if len(sys.argv) == 1 else None)

    export_yolo_to_onnx(args.weights, args.onnx, imgsz=args.imgsz, device=args.device)

    fa = FaceAnalysis(allowed_modules=["detection", "recognition"])
    fa.prepare(ctx_id=0, det_size=(args.imgsz, args.imgsz))
    #fa.prepare(ctx_id=-1, det_size=(args.imgsz, args.imgsz)) # if cpu
    face_gallery = build_face_gallery(args.faces_dir, fa)
    with open(args.face_db, "wb") as f:
        pickle.dump(face_gallery, f)
    print(f"Saved face gallery to {args.face_db}")

    plate_list = build_plate_db_from_list(args.plates_txt)
    with open(args.plates_db, "wb") as f:
        pickle.dump(plate_list, f)
    print(f"Saved plate DB to {args.plates_db}")

if __name__ == "__main__":
    main()