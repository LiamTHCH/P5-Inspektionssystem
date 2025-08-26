import cv2
import numpy as np
import paho.mqtt.client as mqtt
import base64
import json
import time

# MQTT Configuration
MQTT_BROKER = "10.17.2.10"
MQTT_PORT = 1883
MQTT_TOPIC = "spBv1.0/cr/NDATA/rasp/canLidDetector"  # New unique topic

# Define Region of Interest (calibrated for your can's lid position)
LID_ROI = {
    "x": 180,  # Move box more to the right if needed
    "y": 150,  # LOWER this value to move ROI down
    "w": 200,  # Keep width around the can width
    "h": 30    # Height just enough to cover the lid edge
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

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

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

    cap.release()
    cv2.destroyAllWindows()
    client.loop_stop()
    client.disconnect()

if __name__ == "__main__":
    main()
