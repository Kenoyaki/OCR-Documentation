import cv2
import numpy as np
import onnxruntime as ort
import os
from pathlib import Path
import time

# --- CONFIGURATION ---
VIDEO_PATH = r"D:\Kenny\IVS\Face and plate recog\video\Julian.mp4"  # Change this to your video file or 0 for webcam
OUTPUT_DIR = "extracted_faces"         # Folder to save images
ONNX_MODEL = "face_plate_fp16.onnx"       # Path to your YOLO model
CONF_THRESHOLD = 0.6                   # Higher confidence to ensure good quality crops
SAVE_INTERVAL = 5                      # Save a face every N frames (to avoid duplicates)

# --- HELPER FUNCTIONS (Reused from inference.py) ---

def letterbox(img, new_shape=(640,640), color=(114,114,114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
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

class FaceExtractor:
    def __init__(self, onnx_path):
        print(f"Loading model from {onnx_path}...")
        try:
            self.sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except Exception as e:
            print(f"CUDA not available, using CPU. Error: {e}")
            self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        self.input_name = self.sess.get_inputs()[0].name

    def detect(self, img, conf_thres=0.5, iou_thres=0.45):
        # Preprocess
        img_p, ratio, pad = letterbox(img, new_shape=(640, 640))
        img_rgb = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)
        inp = img_rgb.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...]
        inp = inp.astype(np.float16)  # Convert to float16 for the model

        # Inference
        outs = self.sess.run(None, {self.input_name: inp})
        out = outs[0]
        if out.ndim == 3: out = out[0]

        # Handle Output Shape (Transpose if necessary)
        if out.shape[0] < out.shape[1]: 
             out = out.T 
             boxes = xywh2xyxy(out[:, :4])
             cls_scores = out[:, 4:]
             class_ids = np.argmax(cls_scores, axis=1)
             confidences = np.max(cls_scores, axis=1)
        else:
             class_ids = np.argmax(out[:, 5:], axis=1)
             confidences = out[:, 4]
             boxes = xywh2xyxy(out[:, :4])

        # Filter
        mask = confidences > conf_thres
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Rescale
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

def main():
    # Setup paths
    base_dir = Path(__file__).parent
    onnx_path = str(base_dir / ONNX_MODEL)
    output_path = base_dir / OUTPUT_DIR
    output_path.mkdir(exist_ok=True)

    # Initialize
    extractor = FaceExtractor(onnx_path)
    
    # Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    frame_count = 0
    saved_count = 0
    
    print("Starting extraction... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Skip frames to avoid duplicates
        if frame_count % SAVE_INTERVAL != 0:
            continue

        detections = extractor.detect(frame, conf_thres=CONF_THRESHOLD)

        for i, det in enumerate(detections):
            # Class 0 is Face
            if det['class'] == 0:
                x1, y1, x2, y2 = det['box']
                
                # --- NEW CODE: Add Padding (Context) ---
                h, w = frame.shape[:2]
                pad_x = int((x2 - x1) * 0.20) # 20% padding
                pad_y = int((y2 - y1) * 0.20)

                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                # ---------------------------------------

                # Crop
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size > 0:
                    filename = f"face_{frame_count}_{i}.jpg"
                    save_path = output_path / filename
                    cv2.imwrite(str(save_path), face_img)
                    saved_count += 1
                    
                    # Draw on frame for visualization
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show progress
        cv2.putText(frame, f"Saved: {saved_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Extraction", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {saved_count} images to {output_path}")

if __name__ == "__main__":
    main()