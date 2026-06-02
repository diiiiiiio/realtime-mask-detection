"""Mask Detection Server.

Flask backend that serves the static client and exposes /predict + /confirm
routes for running a YOLO mask-detection model against frames captured from a
phone camera.

Run with `python server.py` from any directory; paths are resolved relative
to this file. Set `PORT` env var to change the port (default 5500). HTTPS is
enabled via an ad-hoc self-signed cert so phones can use getUserMedia.
"""

from __future__ import annotations

import base64
import os
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_DIR = os.path.join(BASE_DIR, "..", "Client")
WEIGHTS_PATH = os.path.join(BASE_DIR, "runs", "yolo26mpro.pt")

app = Flask(__name__, static_folder=CLIENT_DIR, static_url_path="")
app.secret_key = "mask-detector-dev"
CORS(app, supports_credentials=True)

print(f"Loading YOLO weights from: {WEIGHTS_PATH}")
model = YOLO(WEIGHTS_PATH)
print(f"Model class names: {model.names}")

# Per-user latest detection cache (session_email -> dict). The image array is
# kept in memory so /confirm can crop without re-running inference.
results_data: dict[str, dict] = {}

# In-memory user store. Wiped on every server restart -- this is fine for a
# demo; swap for a real DB if you ever ship it.
accounts: dict[str, str] = {}


def get_session_email() -> str | None:
    return session.get("current_email")


def decode_base64_image(data_url: str) -> Image.Image:
    """Turn a `data:image/...;base64,xxx` data URL into a PIL Image (RGB)."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(BytesIO(raw)).convert("RGB")


def classify_label(label: str) -> str:
    """Map raw YOLO class name to one of: with_mask / without_mask / incorrect."""
    l = (label or "").lower()
    if "without" in l or "no_mask" in l or "no-mask" in l:
        return "without_mask"
    if "incorrect" in l or "wrong" in l or "weared" in l:
        return "incorrect"
    return "with_mask"


# ---------- Static / auth ----------


@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "login.html")


@app.route("/<path:path>")
def serve_static(path):
    if path == "upload.html" and not get_session_email():
        return send_from_directory(app.static_folder, "login.html")
    return send_from_directory(app.static_folder, path)


@app.route("/login_authorization", methods=["POST"])
def login_authorization():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")
    if email in accounts and accounts[email] == password:
        session["current_email"] = email
        results_data[email] = {"image": None, "boxes": [], "classes": [], "confs": []}
        return jsonify({"success": True})
    if email in accounts:
        return jsonify({"success": False, "error": "Incorrect password"}), 401
    return jsonify({"success": False, "error": "Account does not exist"}), 404


@app.route("/signup_authorization", methods=["POST"])
def signup_authorization():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return jsonify({"success": False, "error": "Email and password required"}), 400
    if email in accounts:
        return jsonify({"success": False, "error": "Email already exists"}), 401
    accounts[email] = password
    return jsonify({"success": True})


@app.route("/logout", methods=["POST"])
def logout():
    email = get_session_email()
    if email:
        session.pop("current_email", None)
        results_data.pop(email, None)
    return jsonify({"success": True})


# ---------- Detection ----------


@app.route("/predict", methods=["POST"])
def predict():
    email = get_session_email()
    if not email:
        return jsonify({"error": "Unauthorized"}), 403

    try:
        data = request.get_json() or {}
        img_data = data.get("image")
        if not img_data:
            return jsonify({"error": "No image provided"}), 400

        pil_image = decode_base64_image(img_data)
        img_rgb = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        prediction = model(img_bgr, imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
        annotated = prediction.plot()

        boxes_t = prediction.boxes
        boxes = boxes_t.xywh.cpu().tolist() if boxes_t is not None else []
        classes = boxes_t.cls.cpu().tolist() if boxes_t is not None else []
        confs = boxes_t.conf.cpu().tolist() if boxes_t is not None else []

        results_data[email] = {
            "image": img_bgr,
            "boxes": boxes,
            "classes": classes,
            "confs": confs,
        }

        summary = {"with_mask": 0, "without_mask": 0, "incorrect": 0}
        for cls_id in classes:
            summary[classify_label(model.names.get(int(cls_id), ""))] += 1

        out_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        out_img.save(buf, format="JPEG", quality=85)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

        print(f"[{email}] detections={len(boxes)} summary={summary}")

        return jsonify({
            "image_with_bboxes": "data:image/jpeg;base64," + encoded,
            "summary": summary,
            "count": len(boxes),
        })
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/predict_realtime", methods=["POST"])
def predict_realtime():
    """Lightweight detection for live mode.

    Returns only the geometry + summary needed to render an overlay client-side.
    Skips the annotated-image render and skips the per-user crop cache, so each
    round-trip is as small/fast as possible.
    """
    email = get_session_email()
    if not email:
        return jsonify({"error": "Unauthorized"}), 403

    try:
        data = request.get_json() or {}
        img_data = data.get("image")
        if not img_data:
            return jsonify({"error": "No image provided"}), 400

        pil_image = decode_base64_image(img_data)
        img_rgb = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = img_bgr.shape[:2]

        prediction = model(img_bgr, imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]

        boxes_t = prediction.boxes
        xywh = boxes_t.xywh.cpu().tolist() if boxes_t is not None else []
        cls_ids = boxes_t.cls.cpu().tolist() if boxes_t is not None else []
        confs = boxes_t.conf.cpu().tolist() if boxes_t is not None else []

        summary = {"with_mask": 0, "without_mask": 0, "incorrect": 0}
        detections = []
        for (x, y, w, h), cls_id, conf in zip(xywh, cls_ids, confs):
            label = model.names.get(int(cls_id), str(int(cls_id)))
            category = classify_label(label)
            summary[category] += 1
            detections.append({
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "label": label,
                "category": category,
                "conf": float(conf),
            })

        return jsonify({
            "detections": detections,
            "summary": summary,
            "count": len(detections),
            "img_w": int(w_img),
            "img_h": int(h_img),
        })
    except Exception as e:
        print("Realtime prediction error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/confirm", methods=["POST"])
def confirm():
    email = get_session_email()
    if not email:
        return jsonify({"error": "Unauthorized"}), 403

    cache = results_data.get(email)
    if not cache or cache.get("image") is None:
        return jsonify({"error": "No detection yet, please capture first"}), 400

    try:
        image = cache["image"]
        boxes = cache["boxes"]
        classes = cache["classes"]
        confs = cache["confs"]
        h_img, w_img = image.shape[:2]

        crops = []
        for idx, (box, cls_id, conf) in enumerate(zip(boxes, classes, confs)):
            x, y, w, h = box
            x1 = max(0, int(x - w / 2))
            y1 = max(0, int(y - h / 2))
            x2 = min(w_img, int(x + w / 2))
            y2 = min(h_img, int(y + h / 2))
            if x2 <= x1 or y2 <= y1:
                continue
            cropped = image[y1:y2, x1:x2]
            ok, buf = cv2.imencode(".jpg", cropped)
            if not ok:
                continue
            encoded = base64.b64encode(buf.tobytes()).decode("utf-8")
            label = model.names.get(int(cls_id), str(int(cls_id)))
            crops.append({
                "id": idx,
                "image": "data:image/jpeg;base64," + encoded,
                "label": label,
                "category": classify_label(label),
                "confidence": float(conf),
            })
        return jsonify({"crops": crops})
    except Exception as e:
        print("Confirm error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5500))
    app.run(host="0.0.0.0", port=port, ssl_context="adhoc")
