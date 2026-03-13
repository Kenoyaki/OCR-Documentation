import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
import easyocr
import pickle
from pathlib import Path
import time
import difflib
from collections import defaultdict, Counter

# --- KONFIGURASI ---
CONF_LOCK_THRESHOLD = 0.50   # Confidence minimal untuk mulai menghitung stability
STABILITY_REQUIRED = 5       # Jumlah frame berturut-turut agar dianggap stabil
MAX_MISSING_FRAMES = 30      # Hapus track jika hilang selama 30 frame
MAX_DIST_TRACK = 60          # Jarak pixel maksimal untuk centroid tracking
track_buffers = defaultdict(lambda: {"embs": [], "ocr": []})
MAX_BUFFER = 8

# --- HELPER FUNCTIONS (Dari Notebook) ---

def letterbox(img, new_shape=(640,640), color=(114,114,114)):
    shape = img.shape[:2] # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def nms_numpy(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0: return []
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return np.array(keep)

def normalize_emb(v):
    norm = np.linalg.norm(v)
    if norm == 0: return v
    return v / norm

def match_face(emb, gallery, threshold=0.35):
    if emb is None or not gallery: return None
    emb = normalize_emb(emb)
    best_score = -1
    best_name = None
    
    for name, g_emb in gallery.items():
        score = np.dot(emb, g_emb) # Cosine similarity
        if score > best_score:
            best_score = score
            best_name = name
            
    # DEBUG: Print score to help adjust threshold
    if best_score > 0.2: 
        print(f"DEBUG: Best match '{best_name}' score: {best_score:.3f}")

    if best_score > threshold:
        return best_name
    return None

def normalize_plate_text(s):
    return "".join(ch for ch in s.upper() if ch.isalnum())

def match_plate(candidate, plate_list, cutoff=0.6):
    if not candidate or not plate_list: return None
    cand = normalize_plate_text(candidate)
    matches = difflib.get_close_matches(cand, plate_list, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def update_track_with_embedding(track_id, emb):
    if emb is None:
        return None
    buf = track_buffers[track_id]["embs"]
    buf.append(normalize_emb(emb))
    if len(buf) > MAX_BUFFER:
        buf.pop(0)
    return normalize_emb(np.mean(buf, axis=0))

def update_track_with_ocr(track_id, ocr_chunks):
    buf = track_buffers[track_id]["ocr"]
    if ocr_chunks:
        top = max(ocr_chunks, key=lambda x: x["conf"])
        txt = normalize_plate_text(top["text"])
        if txt:
            buf.append(txt)
            if len(buf) > MAX_BUFFER:
                buf.pop(0)

    if not buf:
        return None
    return Counter(buf).most_common(1)[0][0]

# --- STATE CLASSES ---

class TrackState:
    def __init__(self):
        self.is_locked = False
        self.label = None
        self.stability_count = 0
        self.last_seen_frame = 0

class SmartCameraSystem:
    def __init__(self, onnx_path, face_db_path, plate_db_path):
        print("Initializing models...")
        
        # 1. Load YOLO ONNX
        try:
            self.sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except Exception as e:
            print(f"Warning: CUDA not available or error loading ONNX ({e}). Using CPU.")
            self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        # 2. Load InsightFace
        self.fa = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self.fa.prepare(ctx_id=0, det_size=(640, 640))
        # self.fa.prepare(ctx_id=-1, det_size=(args.imgsz, args.imgsz)) # if cpu
        
        # 3. Load OCR
        self.ocr = easyocr.Reader(['en'], gpu=True)
        # self.ocr = easyocr.Reader(['en'], gpu=False) # if cpu
        
        # 4. Load Databases
        self.face_gallery = {}
        self.plate_list = []
        
        if Path(face_db_path).exists():
            with open(face_db_path, "rb") as f: self.face_gallery = pickle.load(f)
        else:
            print("Warning: Face DB not found.")

        if Path(plate_db_path).exists():
            with open(plate_db_path, "rb") as f: self.plate_list = pickle.load(f)
        else:
            print("Warning: Plate DB not found.")
        
        # Tracking Variables
        self.track_states = {}
        self.prev_centroids = {}
        self.track_last_seen = {}
        self.next_track_id = 0
        
        print("System Ready.")

    def _infer_yolo(self, img, conf_thres=0.25, iou_thres=0.45):
        # Preprocess
        img_p, ratio, pad = letterbox(img, new_shape=(640, 640))
        img_rgb = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)
        inp = img_rgb.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...] # 1x3x640x640
        # Convert to float16 for fp16 model
        inp = inp.astype(np.float16)

        # Run ONNX
        input_name = self.sess.get_inputs()[0].name
        outs = self.sess.run(None, {input_name: inp})
        
        # Postprocess (Handle output shape)
        out = outs[0] 
        if out.ndim == 3: out = out[0] # Remove batch dim if present
        
        # [x, y, w, h, conf, class_probs...]
        boxes = xywh2xyxy(out[:, :4])
        scores = out[:, 4] # Objectness * Class Conf (simplified for YOLOv8/11 usually)
        
        # Jika output YOLOv8/11 formatnya (Batch, 84, 8400), perlu transpose
        # Tapi asumsi ini menggunakan export default ultralytics yang sudah simplified
        # Jika outputnya [4+cls, N], kita perlu transpose. 
        # Cek shape:
        if out.shape[0] < out.shape[1]: # e.g. (84, 8400)
             out = out.T # Jadi (8400, 84)
             boxes = xywh2xyxy(out[:, :4])
             # Class scores start at index 4
             cls_scores = out[:, 4:]
             class_ids = np.argmax(cls_scores, axis=1)
             confidences = np.max(cls_scores, axis=1)
        else:
             # Fallback logic (tergantung versi export)
             class_ids = np.argmax(out[:, 5:], axis=1)
             confidences = out[:, 4]

        # Filter by confidence
        mask = confidences > conf_thres
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        # Rescale boxes to original image
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes /= ratio
        
        # NMS
        indices = nms_numpy(boxes, confidences, iou_thres)
        
        results = []
        for i in indices:
            results.append({
                "box": boxes[i].astype(int),
                "score": confidences[i],
                "class": class_ids[i]
            })
        return results

    def _update_tracker(self, box, frame_idx):
        # Simple Centroid Tracking
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        centroid = (cx, cy)

        best_id = None
        min_dist = 1e9

        # Cari track ID terdekat
        for tid, prev_c in self.prev_centroids.items():
            dist = np.hypot(prev_c[0] - cx, prev_c[1] - cy)
            if dist < min_dist:
                min_dist = dist
                best_id = tid

        if best_id is not None and min_dist < MAX_DIST_TRACK:
            # Update existing track
            self.prev_centroids[best_id] = centroid
            self.track_last_seen[best_id] = frame_idx
            return best_id
        else:
            # Create new track
            new_id = self.next_track_id
            self.next_track_id += 1
            self.prev_centroids[new_id] = centroid
            self.track_last_seen[new_id] = frame_idx
            return new_id

    def process_stream(self, source=0):
        # If source is an RTSP URL, prefer TCP transport to avoid UDP packet loss
        # Append ?tcp or &tcp when missing (simple, non-invasive fix)
        if isinstance(source, str) and source.lower().startswith("rtsp://"):
            if "?tcp" not in source and "&tcp" not in source:
                source = source + ("?tcp" if "?" not in source else "&tcp")
            print(f"Using RTSP over TCP: {source}")

        # Try to open with FFMPEG backend first (better RTSP support), fall back to default
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(source)
        except Exception:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError(f"Cannot open camera source: {source}")

        # Reduce internal buffer to minimize latency (if backend supports it)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Create resizable window and compute display scale to fit screen
        cv2.namedWindow("Smart Stream", cv2.WINDOW_NORMAL)
        try:
            import ctypes
            screen_w = ctypes.windll.user32.GetSystemMetrics(0)
            screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        except Exception:
            screen_w, screen_h = 1366, 768
        MAX_DISPLAY_W, MAX_DISPLAY_H = int(screen_w * 0.9), int(screen_h * 0.85)
        def scale_frame_for_display(f):
            h, w = f.shape[:2]
            scale = min(1.0, MAX_DISPLAY_W / w, MAX_DISPLAY_H / h)
            if scale < 1.0:
                return cv2.resize(f, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            return f

        # reconnect parameters
        RECONNECT_DELAY = 1.0
        RECONNECT_ATTEMPTS = 5
        reconnect_tries = 0

        frame_idx = 0
        fps_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Attempt reconnect instead of hard exit
                    reconnect_tries += 1
                    print(f"Warning: frame read failed (try {reconnect_tries}/{RECONNECT_ATTEMPTS})")
                    cap.release()
                    time.sleep(RECONNECT_DELAY)
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG) if hasattr(cv2, 'CAP_FFMPEG') else cv2.VideoCapture(source)
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    if reconnect_tries > RECONNECT_ATTEMPTS:
                        print("Stream ended (reconnect attempts exhausted).")
                        break
                    continue
                reconnect_tries = 0                
                
                frame_idx += 1
                
                # 1. Detect (YOLO)
                detections = self._infer_yolo(frame)
                
                # 2. Process Each Detection
                for det in detections:
                    box = det['box']
                    cls_id = det['class']
                    conf = det['score']
                    
                    # Track
                    track_id = self._update_tracker(box, frame_idx)
                    
                    # Init State
                    if track_id not in self.track_states:
                        self.track_states[track_id] = TrackState()
                    state = self.track_states[track_id]
                    state.last_seen_frame = frame_idx

                    # --- LOGIC RECOGNITION ---
                    if not state.is_locked:
                        # Check Stability
                        if conf > CONF_LOCK_THRESHOLD:
                            state.stability_count += 1
                        else:
                            state.stability_count = 0
                        
                        # Trigger Recognition if Stable
                        if state.stability_count >= STABILITY_REQUIRED:
                            x1, y1, x2, y2 = box
                            h, w = frame.shape[:2]
                            # Clamp
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            if x2 > x1 and y2 > y1:
                                roi = frame[y1:y2, x1:x2]
                                found_label = None
                                
                                # FACE
                                if cls_id == 0:
                                    # --- FIX: Add padding for InsightFace ---
                                    # InsightFace needs context (hair, chin, ears) to detect landmarks correctly.
                                    # A tight box from YOLO might cut these off.
                                    pad_x = int((x2 - x1) * 0.20) # 20% padding
                                    pad_y = int((y2 - y1) * 0.20)
                                    
                                    x1_p = max(0, x1 - pad_x)
                                    y1_p = max(0, y1 - pad_y)
                                    x2_p = min(w, x2 + pad_x)
                                    y2_p = min(h, y2 + pad_y)
                                    
                                    roi_padded = frame[y1_p:y2_p, x1_p:x2_p]
                                    rgb_roi = cv2.cvtColor(roi_padded, cv2.COLOR_BGR2RGB)
                                    
                                    faces = self.fa.get(rgb_roi)
                                    
                                    if not faces:
                                        print(f"DEBUG: ID {track_id} Frame {frame_idx} - InsightFace failed. Crop size: {roi_padded.shape}")
                                    else:
                                        print(f"DEBUG: ID {track_id} Frame {frame_idx} - InsightFace detected {len(faces)} face(s)")
                                        # Pick the largest face in the crop (in case background faces are caught)
                                        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                                        
                                        emb = getattr(faces[0], "embedding", None)
                                        print(f"DEBUG: ID {track_id} - Embedding shape: {emb.shape if emb is not None else 'None'}")
                                        avg_emb = update_track_with_embedding(track_id, emb)
                                        found_label = match_face(avg_emb, self.face_gallery, threshold=0.30)
                                
                                # PLATE
                                elif cls_id == 1:
                                    ocr_res = self.ocr.readtext(roi, detail=1)
                                    chunks = [
                                        {"text": normalize_plate_text(text), "conf": conf}
                                        for _, text, conf in ocr_res
                                        if 5 <= len(normalize_plate_text(text)) <= 10
                                    ]
                                    vote = update_track_with_ocr(track_id, chunks)
                                    if vote:
                                        found_label = match_plate(vote, self.plate_list) or vote

                                if found_label:
                                    state.is_locked = True
                                    state.label = found_label
                                    print(f"Locked ID {track_id}: {found_label}")

                    # --- DRAWING ---
                    x1, y1, x2, y2 = box
                    if state.is_locked:
                        color = (0, 255, 0) # Green
                        text = f"ID:{track_id} | {state.label}"
                    elif state.stability_count > 0:
                        color = (0, 255, 255) # Yellow
                        text = f"ID:{track_id} | ..."
                    else:
                        color = (255, 255, 255) # White
                        text = f"ID:{track_id}"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 3. Cleanup Old Tracks
                stale_ids = [tid for tid, last in self.track_last_seen.items() 
                             if frame_idx - last > MAX_MISSING_FRAMES]
                for tid in stale_ids:
                    self.prev_centroids.pop(tid, None)
                    self.track_last_seen.pop(tid, None)
                    self.track_states.pop(tid, None)

                # FPS
                if frame_idx % 10 == 0:
                    fps = 10.0 / (time.time() - fps_time)
                    fps_time = time.time()
                    print(f"FPS: {fps:.1f}")

                disp = scale_frame_for_display(frame)
                cv2.imshow("Smart Stream", disp)
                if cv2.waitKey(1) & 0xFF == 27: # ESC
                    break
                    
        except Exception as e:
            print(f"Runtime Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Konfigurasi Path (Gunakan absolute path agar aman saat dijalankan dari mana saja)
    BASE_DIR = Path(__file__).parent
    
    # Pastikan file-file ini ada di folder yang sama atau sesuaikan path-nya
    app = SmartCameraSystem(
        onnx_path=str(BASE_DIR / "face_plate_fp16.onnx"),
        face_db_path=str(BASE_DIR / "faces.pkl"),
        plate_db_path=str(BASE_DIR / "plates.pkl")
    )
   
    #app.process_stream("rtsp://192.168.8.177:5543/a66abb5684c45962d887564f08346e8d/live/channel0")
    app.process_stream(0)