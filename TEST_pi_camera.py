import cv2
import numpy as np
import paho.mqtt.client as mqtt
import base64
import json
import time
from picamera2 import Picamera2

from dotenv import load_dotenv
import os

load_dotenv()

MQTT_BROKER = os.getenv('MQTT_BROKER')
MQTT_PORT = int(os.getenv('MQTT_PORT', '1883'))  # Default port 1883
MQTT_TOPIC = os.getenv('MQTT_TOPIC')

# Region of Interest (calibrated for your can's lid position)
LID_ROI = {
    "x": 180,
    "y": 150,
    "w": 200,
    "h": 30
}

def detect_lid_fixed_position(frame):
    x, y, w, h = LID_ROI.values()
    lid_roi = frame[y:y+h, x:x+w]

    gray = cv2.cvtColor(lid_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Enhance edges with dilation
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    edge_pixel_count = cv2.countNonZero(edges)
    total_pixels = w * h
    edge_ratio = edge_pixel_count / total_pixels

    # Mean intensity to assist detection for transparent lid
    mean_intensity = np.mean(gray)

    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(frame, f"Edge Ratio: {edge_ratio:.2f}", (x, y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Mean Intensity: {mean_intensity:.1f}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Combined threshold for detection - tune thresholds as needed
    lid_detected = (edge_ratio > 0.02) and (mean_intensity < 180)

    return lid_detected, frame, lid_roi

def encode_image_to_base64(image):
    ret, jpeg = cv2.imencode('.jpg', image)
    if not ret:
        return None
    return base64.b64encode(jpeg.tobytes()).decode('utf-8')

def publish_mqtt(client, lid_status, image_b64):
    # Convert numpy.bool_ to native bool
    lid_status = bool(lid_status)

    payload = {
        "timestamp": int(time.time() * 1000),
        "metrics": [
            {
                "name": "LidStatus",
                "type": "Boolean",
                "value": lid_status
            },
            {
                "name": "LidImage",
                "type": "String",
                "value": image_b64
            }
        ]
    }
    payload_str = json.dumps(payload)
    client.publish(MQTT_TOPIC, payload_str)

def main():
    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()
    time.sleep(2)  # Camera warm-up

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        lid_detected, result_frame, lid_roi = detect_lid_fixed_position(frame)

        status_text = "LID OK" if lid_detected else "LID MISSING"
        color = (0, 255, 0) if lid_detected else (0, 0, 255)
        cv2.putText(result_frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        image_b64 = encode_image_to_base64(lid_roi)
        publish_mqtt(client, lid_detected, image_b64)

        cv2.imshow("Lid Detection - Fixed ROI", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()
    client.loop_stop()
    client.disconnect()

if __name__ == "__main__":
    main()
