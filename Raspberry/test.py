import cv2
import numpy as np

# Define Region of Interest (calibrated for your can's lid position)
# You should manually calibrate these values using a test image.
LID_ROI = {
    "x": 180,   # Move box more to the right if needed
    "y": 150,   # LOWER this value to move ROI down
    "w": 200,   # Keep width around the can width
    "h": 30     # Height just enough to cover the lid edge
}


def detect_lid_fixed_position(frame):
    # Crop to the lid area only
    x, y, w, h = LID_ROI.values()
    lid_roi = frame[y:y+h, x:x+w]
    
    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(lid_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Count edge pixels — we expect strong edge signal if lid is present
    edge_pixel_count = cv2.countNonZero(edges)
    total_pixels = w * h
    edge_ratio = edge_pixel_count / total_pixels

    # Visual overlay
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(frame, f"Edge Ratio: {edge_ratio:.2f}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Decision threshold — tune this for your setup
    lid_detected = edge_ratio > 0.02  # Adjust this threshold based on real tests

    return lid_detected, frame

def main():
    cap = cv2.VideoCapture(0)  # Adjust camera index if needed

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        lid_detected, result_frame = detect_lid_fixed_position(frame)

        # Display detection status
        status_text = "LID OK" if lid_detected else "LID MISSING"
        color = (0, 255, 0) if lid_detected else (0, 0, 255)
        cv2.putText(result_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show result
        cv2.imshow("Lid Detection - Fixed ROI", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
