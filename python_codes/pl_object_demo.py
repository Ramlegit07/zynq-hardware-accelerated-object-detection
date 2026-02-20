from flask import Flask, Response
from pynq import Overlay, allocate
import cv2
import numpy as np
import threading
import time

app = Flask(__name__)

# ==============================
# LOAD FPGA OVERLAY
# ==============================
ol = Overlay("design_1.bit")
conv = ol.conv2d_hls_0

# Allocate buffers for PL conv
input_buf = allocate(shape=(9,), dtype=np.float32)
kernel_buf = allocate(shape=(9,), dtype=np.float32)
output_buf = allocate(shape=(1,), dtype=np.float32)

# Simple sharpening kernel
kernel_values = np.array([
    -1, -1, -1,
    -1,  9, -1,
    -1, -1, -1
], dtype=np.float32)

kernel_buf[:] = kernel_values

# ==============================
# LOAD OBJECT DETECTION MODEL (PS)
# ==============================
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "mobilenet_iter_73000.caffemodel"
)

CLASSES = [
    "background","aeroplane","bicycle","bird","boat",
    "bottle","bus","car","cat","chair","cow","diningtable",
    "dog","horse","motorbike","person","pottedplant","sheep",
    "sofa","train","tvmonitor"
]

# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not detected")
    exit()

latest_frame = None
frame_lock = threading.Lock()

last_boxes = []
last_labels = []
cpu_latency = 0
pl_latency = 0

# ==============================
# DETECTION THREAD (PS)
# ==============================
def detection_thread():
    global last_boxes, last_labels, cpu_latency

    while True:
        time.sleep(0.5)  # run detection every 0.5s

        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        start = time.perf_counter()

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (224, 224)),
            0.007843, (224, 224), 127.5
        )

        net.setInput(blob)
        detections = net.forward()

        h, w = frame.shape[:2]
        boxes = []
        labels = []

        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.5:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                boxes.append((x1, y1, x2, y2))
                labels.append(CLASSES[idx])

        end = time.perf_counter()
        cpu_latency = (end - start) * 1000

        last_boxes = boxes
        last_labels = labels

threading.Thread(target=detection_thread, daemon=True).start()

# ==============================
# STREAM LOOP
# ==============================
def generate():
    global latest_frame, pl_latency

    fps_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        with frame_lock:
            latest_frame = frame.copy()

        # -------------------------
        # RUN PL CONVOLUTION
        # -------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cx, cy = h // 2, w // 2
        patch = gray[cx-1:cx+2, cy-1:cy+2]

        patch = patch.astype(np.float32).flatten()
        input_buf[:] = patch

        start = time.perf_counter()

        conv.write(0x10, input_buf.physical_address)
        conv.write(0x18, kernel_buf.physical_address)
        conv.write(0x20, output_buf.physical_address)
        conv.write(0x00, 1)

        while (conv.read(0x00) & 0x2) == 0:
            pass

        _ = output_buf[0]

        end = time.perf_counter()
        pl_latency = (end - start) * 1000

        # -------------------------
        # DRAW DETECTIONS
        # -------------------------
        for (box, label) in zip(last_boxes, last_labels):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0,255,0), 2)
            cv2.putText(frame, label,
                        (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,0),2)

        # -------------------------
        # FPS
        # -------------------------
        frame_count += 1
        if time.time() - fps_time >= 1:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()

        # -------------------------
        # OVERLAY INFO
        # -------------------------
        cv2.putText(frame, f"Stream FPS: {fps}",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,255,0),2)

        cv2.putText(frame, f"CPU Latency: {cpu_latency:.1f} ms",
                    (20,70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,255,0),2)

        cv2.putText(frame, f"PL Latency: {pl_latency:.3f} ms",
                    (20,100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,255,0),2)

        ret, buffer = cv2.imencode(".jpg", frame)

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

@app.route("/")
def video():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("Open browser: http://10.11.41.26:5000")
    app.run(host="0.0.0.0", port=5000)