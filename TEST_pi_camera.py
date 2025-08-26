import cv2
import numpy as np
import paho.mqtt.client as mqtt
import base64
import json
import time
from picamera2 import Picamera2
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# MQTT Config
MQTT_BROKER = os.getenv('MQTT_BROKER')
MQTT_PORT = int(os.getenv('MQTT_PORT', '1883'))
MQTT_TOPIC = os.getenv('MQTT_TOPIC')

# Debug flag
DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# ROI and thresholds from environment or defaults
LID_ROI = {
    "x": int(os.getenv('LID_ROI_X', '180')),
    "y": int(os.getenv('LID_ROI_Y', '150')),
    "w": int(os.getenv('LID_ROI_W', '200')),
    "h": int(os.getenv('LID_ROI_H', '30'))
}

EDGE_RATIO_THRESHOLD = float(os.getenv('EDGE_RATIO_THRESHOLD', '0.02'))
MEAN_INTENSITY_THRESHOLD = float(os.getenv('MEAN_INTENSITY_THRESHOLD', '180'))

def detect_lid_fixed_position(frame):
    x, y, w, h = LID_ROI.values()
    lid_roi = frame[y:y+h, x:x+w]

    gray = cv2.cvtColor(lid_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    edge_pixel_count = cv2.countNonZero(edges)
    total_pixels = w * h
    edge_ratio = edge_pixel_count / total_pixels

    mean_intensity = np.mean(gray)

    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(frame, f"Edge Ratio: {edge_ratio:.2f}", (x, y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Mean Intensity: {mean_intensity:.1f}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    lid_detected = (edge_ratio > EDGE_RATIO_THRESHOLD) and (mean_intensity < MEAN_INTENSITY_THRESHOLD)

    if DEBUG:
        logging.debug(f"Edge pixel count: {edge_pixel_count}")
        logging.debug(f"Total pixels in ROI: {total_pixels}")
        logging.debug(f"Edge ratio: {edge_ratio}")
        logging.debug(f"Mean intensity: {mean_intensity}")
        logging.debug(f"Lid detected: {lid_detected}")

    return lid_detected, frame, lid_roi

def encode_image_to_base64(image):
    ret, jpeg = cv2.imencode('.jpg', image)
    if not ret:
        logging.error("Failed to encode image to JPEG")
        return None
    return base64.b64encode(jpeg.tobytes()).decode('utf-8')

def publish_mqtt(client, lid_status, image_b64):
    lid_status = bool(lid_status)  # Convert numpy.bool_ to native bool

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
    try:
        payload_str = json.dumps(payload)
    except Exception as e:
        logging.error(f"Failed to encode JSON payload: {e}")
        return

    result = client.publish(MQTT_TOPIC, payload_str)
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        logging.info(f"Published lid status={lid_status} to topic {MQTT_TOPIC}")
    else:
        logging.error(f"Failed to publish MQTT message, error code: {result.rc}")

# MQTT callbacks

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logging.info("Connected successfully to MQTT Broker")
    else:
        logging.error(f"Failed to connect, return code {rc}")

def on_disconnect(client, userdata, rc):
    logging.warning(f"Disconnected from MQTT Broker with code {rc}")

def on_publish(client, userdata, mid):
    logging.debug(f"Message published with mid: {mid}")

def on_log(client, userdata, level, buf):
    if DEBUG:
        logging.debug(f"MQTT log: {buf}")

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish = on_publish
    client.on_log = on_log

    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()
    time.sleep(2)  # camera warm-up

    STABLE_TIME_SECONDS = 5
    last_lid_status = None
    last_status_change_time = time.time()
    message_sent_time = 0

    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            lid_detected, result_frame, lid_roi = detect_lid_fixed_position(frame)

            if lid_detected != last_lid_status:
                last_lid_status = lid_detected
                last_status_change_time = time.time()
                if DEBUG:
                    logging.debug(f"Lid status changed to {lid_detected}, waiting for stability...")

            elif (time.time() - last_status_change_time) >= STABLE_TIME_SECONDS and message_sent_time < last_status_change_time:
                image_b64 = encode_image_to_base64(lid_roi)
                publish_mqtt(client, lid_detected, image_b64)
                message_sent_time = time.time()

            status_text = "LID OK" if lid_detected else "LID MISSING"
            color = (0, 255, 0) if lid_detected else (0, 0, 255)
            cv2.putText(result_frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("Lid Detection - Fixed ROI", result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logging.info("Program interrupted by user")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
