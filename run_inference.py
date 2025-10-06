import os
import torch
import numpy as np
import cv2
from PIL import Image
import pickle
from collections import deque
import json

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model_weights", "patchcore_model.pkl")
TEST_DIR = os.path.join(BASE_DIR, "test_image")
OUT_DIR = os.path.join(BASE_DIR, "output_image")
SEGMENTED_DIR = os.path.join(BASE_DIR, "labeled_segmented")
ANNOTATION_DIR = os.path.join(BASE_DIR, "annotations_json")

# Create output directories
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SEGMENTED_DIR, exist_ok=True)
os.makedirs(ANNOTATION_DIR, exist_ok=True)

# -----------------------------
# Calibration parameter
# -----------------------------
# PERCENT_THRESHOLD = how sensitive you want (like your bar 0–100%)
# Smaller % → more sensitive, Larger % → less sensitive
PERCENT_THRESHOLD = 50   # e.g., detect at 50% temperature increase

# Convert percent into k value in range [1.1, 2.1]
def percent_to_k(percent):
    # Clamp input between 0 and 100
    percent = max(0, min(percent, 100))
    # Map 0% → 1.1 (very sensitive), 100% → 2.1 (least sensitive)
    return 1.1 + (percent / 100.0) * (2.1 - 1.1)

k_value = percent_to_k(PERCENT_THRESHOLD)

k = k_value

# -------------------------
# Load model
# -------------------------
print("🔹 Loading model from saved file...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)
model.eval()
model = model.to(device)
print("✅ Model loaded for inference.")

# -------------------------
# Heuristic anomaly classification function
# -------------------------
def classify_anomalies(filtered_img, anomaly_map=None):
    hsv = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2HSV)
    h, w = filtered_img.shape[:2]
    total_area = float(w * h)

    # Adaptive threshold for anomaly map if provided
    if anomaly_map is not None:
        thresh = anomaly_map.mean() + k*anomaly_map.std()
        bin_mask = anomaly_map > thresh
    else:
        bin_mask = np.zeros((h, w), dtype=bool)

    # Warm mask (thermal hotspot rule)
    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            H, S, V = hsv[y, x]
            hC, sC, vC = H/180.0, S/255.0, V/255.0
            warm_hue = (hC <= 0.17) or (hC >= 0.95)
            warm_sat = sC >= 0.35
            warm_val = vC >= 0.5
            if warm_hue and warm_sat and warm_val:
                mask[y, x] = 1

    # -------------------------
    # Ignore right-side FLIR bar and thin bars
    sidebar_width = int(w * 0.10)  # right 10% width
    mask[:, w - sidebar_width : w] = 0

    max_check_width = max(1, int(w*0.06))
    min_check_width = max(1, int(w*0.005))
    hsv_float = hsv.astype(np.float32)

    for cand_w in range(min_check_width, max_check_width+1):
        x0 = w - cand_w
        region = hsv_float[:, x0:w, :]
        hue = region[...,0]
        sat = region[...,1]
        val = region[...,2]
        hue_var = np.mean(np.std(hue, axis=0))
        sat_mean = np.mean(sat)
        val_mean = np.mean(val)
        if sat_mean > 40 and val_mean > 120 and hue_var < 8:
            mask[:, x0:w] = 0
            break
    # -------------------------

    # Connected components → boxes
    visited = np.zeros_like(mask, dtype=bool)
    boxes = []
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    min_area = max(32, int(w*h*0.001))

    for y in range(h):
        for x in range(w):
            if mask[y, x] and not visited[y, x]:
                q = deque([(x,y)])
                visited[y, x] = True
                minX = maxX = x
                minY = maxY = y
                area = 0
                while q:
                    px, py = q.popleft()
                    area += 1
                    minX, maxX = min(minX, px), max(maxX, px)
                    minY, maxY = min(minY, py), max(maxY, py)
                    for dx, dy in dirs:
                        nx, ny = px+dx, py+dy
                        if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((nx, ny))
                if area >= min_area:
                    boxes.append((minX, minY, maxX-minX+1, maxY-minY+1))

    # Classify boxes and compute confidence
    labels = []
    confidences = []
    severities = []  # 'Faulty' or 'Potentially Faulty'
    for (x,y,bw,bh) in boxes:
        area_frac = (bw*bh)/total_area
        aspect = max(bw, bh)/max(1.0, min(bw, bh))
        center_x0, center_y0 = int(w*0.33), int(h*0.33)
        center_x1, center_y1 = int(w*0.67), int(h*0.67)
        ox0, oy0 = max(x, center_x0), max(y, center_y0)
        ox1, oy1 = min(x+bw, center_x1), min(y+bh, center_y1)
        overlap = max(0, ox1-ox0) * max(0, oy1-oy0)
        overlap_frac = overlap/(bw*bh)

        box_hsv = hsv[y:y+bh, x:x+bw, :].astype(np.float32)
        H = box_hsv[...,0]  # 0..180
        S = box_hsv[...,1]  # 0..255
        V = box_hsv[...,2]  # 0..255
        # Define color classes
        red_mask = ((H <= 10) | (H >= 160)) & (S >= 100) & (V >= 100)
        orange_mask = (H > 10) & (H <= 25) & (S >= 100) & (V >= 100)
        yellow_mask = (H > 25) & (H <= 35) & (S >= 100) & (V >= 100)
        warm_mask_local = red_mask | orange_mask | yellow_mask
        warm_count_local = np.count_nonzero(warm_mask_local)
        if warm_count_local > 0:
            red_orange_frac = (np.count_nonzero(red_mask | orange_mask))/float(warm_count_local)
            yellow_frac_local = (np.count_nonzero(yellow_mask))/float(warm_count_local)
        else:
            red_orange_frac = 0.0
            yellow_frac_local = 0.0

        v_mean = float(np.mean(V/255.0))

        # Base category per existing geometry rules
        if area_frac >= 0.10 and (overlap_frac >= 0.4 or area_frac >= 0.30):
            base_label = "Loose Joint"
            # Severity by color: red/orange → Faulty; yellow only → Potentially Faulty
            severity = "Faulty" if red_orange_frac >= 0.5 else "Potentially Faulty"
            labels.append(f"{base_label} ({severity})")
            severities.append(severity)
            confidences.append(min(1.0, 0.6 + 0.8*area_frac))
        elif aspect >= 2.0:
            # Distinguish full wire overload (potential) vs point on wire
            if area_frac >= 0.30 and (yellow_frac_local >= red_orange_frac):
                base_label = "Full Wire Overload"
                severity = "Potentially Faulty"
            else:
                # Small region on wire: decide by color
                base_label = "Point Overload"
                severity = "Faulty" if red_orange_frac >= 0.5 else "Potentially Faulty"
            labels.append(f"{base_label} ({severity})")
            severities.append(severity)
            confidences.append(min(1.0, 0.5 + 0.2*aspect))
        else:
            base_label = "Point Overload"
            severity = "Faulty" if red_orange_frac >= 0.5 else "Potentially Faulty"
            labels.append(f"{base_label} ({severity})")
            severities.append(severity)
            confidences.append(min(1.0, 0.5 + 0.5*v_mean))

    if not boxes:
        return "Normal", [], [], [], []

    return None, boxes, labels, confidences, severities

# -------------------------
# Process all images in test folder
# -------------------------
for img_file in os.listdir(TEST_DIR):
    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    TEST_IMAGE = os.path.join(TEST_DIR, img_file)
    OUT_PATH = os.path.join(OUT_DIR, img_file.rsplit('.',1)[0]+"_labeled.png")
    SEGMENTED_PATH = os.path.join(SEGMENTED_DIR, img_file.rsplit('.',1)[0]+"_segmented.png")
    ANNOTATION_PATH = os.path.join(ANNOTATION_DIR, img_file.rsplit('.',1)[0]+'.json')

    # Load image
    img = Image.open(TEST_IMAGE).convert("RGB")
    img_tensor = torch.tensor(np.array(img)).permute(2,0,1).unsqueeze(0).float()/255.0
    img_tensor = img_tensor.to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        if hasattr(output, 'anomaly_map'):
            anomaly_map = output.anomaly_map.squeeze().cpu().numpy()
        elif isinstance(output, (tuple, list)) and len(output) > 1:
            anomaly_map = output[1].squeeze().cpu().numpy()
        else:
            anomaly_map = None

    # Post-processing
    if anomaly_map is not None:
        norm_map = (255 * (anomaly_map - anomaly_map.min()) / (np.ptp(anomaly_map) + 1e-8)).astype(np.uint8)
        mask_img = Image.fromarray(norm_map).resize(img.size, resample=Image.BILINEAR)
        bin_mask = np.array(mask_img) > 128
    else:
        bin_mask = np.zeros_like(np.array(img)[:,:,0], dtype=bool)

    orig_np = np.array(img)
    filtered_img = np.zeros_like(orig_np)
    filtered_img[bin_mask] = orig_np[bin_mask]

    # Classify anomalies with confidence
    image_label, box_list, label_list, conf_list, severities = classify_anomalies(filtered_img, anomaly_map=anomaly_map)

    # Save segmented labeled image
    segmented_img_with_labels = filtered_img.copy()
    for (x,y,wb,hb), l, conf, sev in zip(box_list, label_list, conf_list, severities):
        # Red for Faulty, Yellow for Potentially Faulty
        if sev.lower().startswith('faulty'):
            color_box = (0,0,255)      # red
            color_text = (255,255,255) # white for contrast
        else:
            color_box = (0,255,255)    # yellow
            color_text = (0,0,0)       # black for contrast
        cv2.rectangle(segmented_img_with_labels, (x,y), (x+wb,y+hb), color_box, 2)
        cv2.putText(segmented_img_with_labels, f"{l} {conf:.2f}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)

    if not box_list:
        cv2.putText(segmented_img_with_labels,"Normal",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    cv2.imwrite(SEGMENTED_PATH, cv2.cvtColor(segmented_img_with_labels, cv2.COLOR_RGB2BGR))

    # Save final labeled image on original background
    final_output_img = orig_np.copy()
    for (x,y,wb,hb), l, conf, sev in zip(box_list, label_list, conf_list, severities):
        if sev.lower().startswith('faulty'):
            color_box = (0,0,255)
            color_text = (255,255,255)
        else:
            color_box = (0,255,255)
            color_text = (0,0,0)
        cv2.rectangle(final_output_img, (x,y), (x+wb,y+hb), color_box, 2)
        cv2.putText(final_output_img, f"{l} {conf:.2f}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)

    if not box_list:
        cv2.putText(final_output_img,"Normal",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    cv2.imwrite(OUT_PATH, cv2.cvtColor(final_output_img, cv2.COLOR_RGB2BGR))

    # -------------------------
    # JSON Annotation Export
    # -------------------------
    annotation = {
        "image": img_file,
        "status": "Normal" if not box_list else "Anomalies",
        "anomalies": [],
    }

    for (x,y,wb,hb), l, conf, sev in zip(box_list,label_list,conf_list, severities):
        lname = l.lower()
        if "loose" in lname:
            category = "loose_joint"
        elif "wire" in lname:
            category = "wire_overload"
        elif "point" in lname:
            category = "point_overload"
        else:
            category = "anomaly"

        annotation["anomalies"].append({
            "label": l,
            "category": category,
            "severity": sev,
            "confidence": float(conf),
            "bbox": {"x": int(x), "y": int(y), "width": int(wb), "height": int(hb)}
        })

    json_path = ANNOTATION_PATH
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(annotation, jf, indent=2)

    print(f"✅ Processed {img_file}:")
    print(f"   - Segmented: {SEGMENTED_PATH}")
    print(f"   - Complete:  {OUT_PATH}")
    print(f"   - Annotation: {json_path}")

print("🎉 All images processed successfully.")
