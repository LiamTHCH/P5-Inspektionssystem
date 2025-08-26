import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('20240327_140909.jpg')


def calculate_darkness(roi):
    # Calculate darkness value as the mean pixel intensity
    darkness = np.mean(roi)
    return darkness


# Get image dimensions
height, width = image.shape[:2]

# Define the region of interest (upper half of the image)
roi = image[0:height//2, :]
cv2.imwrite('image.jpg', image)
#roi = image
cv2.imwrite('roi.jpg', roi)

# Convert ROI to grayscale
try:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
except:
    gray = roi

cv2.imwrite('gray.jpg', roi)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blurred, 50, 150)

# Perform Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Filter and draw horizontal lines
if lines is not None:
    # Define tolerance range (adjustable)
    tolerance_cm = 10
    tolerance_pixels = int(tolerance_cm * 10)  # 10 pixels per cm
    # Dictionary to store lines grouped by y-coordinate
    lines_by_y = {}
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate slope
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
        # Threshold for horizontal lines (adjust as needed)
        if abs(slope) < 0.1:
            # Group lines by y-coordinate within tolerance
            y = (y1 + y2) // 2
            for dy in range(-tolerance_pixels, tolerance_pixels + 1):
                y_key = y + dy
                if y_key in lines_by_y:
                    lines_by_y[y_key].append(line[0])
                    break  # Add the line to the first matching y-coordinate
            else:
                lines_by_y[y] = [line[0]]

    # Draw continuous lines for each group of lines at the same height
    nb_big_lines = 0
    big_lignes = []
    for y, group_lines in lines_by_y.items():
        # Find minimum and maximum x-coordinates
        min_x = min(group_lines, key=lambda x: x[0])[0]
        max_x = max(group_lines, key=lambda x: x[2])[2]
        # Draw a single line covering all lines at this height
        if nb_big_lines < 2:
            # group 1
            cv2.line(roi, (min_x, y), (max_x, y), (0, 0, 255), 2)
            big_lignes.append({"y": y, "group": 1, "min_x": min_x, "max_x": max_x})
            nb_big_lines += 1
        elif 2 <= nb_big_lines <= 4:
            # group 2
            cv2.line(roi, (min_x, y), (max_x, y), (0, 255, 0), 2)
            big_lignes.append({"y": y, "group": 2, "min_x": min_x, "max_x": max_x})
            nb_big_lines += 1
        else:
            # group 3
            cv2.line(roi, (min_x, y), (max_x, y), (0, 0, 255), 2)
            big_lignes.append({"y": y, "group": 3, "min_x": min_x, "max_x": max_x})
            nb_big_lines += 1

        print(f"Horizontal line detected at y = {y}")

for i in range(len(big_lignes)):
    if i == 0:
        smallest_min_x = min(big_lignes[i]["min_x"], big_lignes[i+1]["min_x"])
        cv2.rectangle(roi, (smallest_min_x, big_lignes[i]["y"]), (big_lignes[i]["max_x"], big_lignes[i+1]["y"]), (0, 0, 255), 2)
        roi2 = roi
    if i == 2:
        smallest_min_x = min(big_lignes[i]["min_x"], big_lignes[i+1]["min_x"])
        cv2.rectangle(roi, (smallest_min_x, big_lignes[i]["y"]), (big_lignes[i]["max_x"], big_lignes[i+1]["y"]), (0, 255, 0), 2)
        roi2 = image[big_lignes[i+1]["y"]:big_lignes[i]["y"], smallest_min_x:big_lignes[i]["max_x"]]

try:
    roi_gray = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.equalizeHist(roi_gray)
except:
    roi_gray = roi2
    enhanced_gray = roi_gray

_, thresholded = cv2.threshold(enhanced_gray, 50, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = roi2.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)



# Convert ROI from BGR to RGB for displaying with matplotlib
try:
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi2_rgb = cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB)

except:
        pass
roi_rgb = roi
roi2_rgb = roi2
cv2.imwrite('roi2.jpg', roi2)
cv2.imwrite('thresholded.jpg', thresholded)
cv2.imwrite('contour_image.jpg', contour_image)
# Display the result using matplotlib
fig, ax = plt.subplots(1, 4, figsize=(15, 7))
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].axis('off')
ax[0].set_title('Original Image')
ax[1].imshow(roi_rgb)
ax[1].axis('off')
ax[1].set_title('ROI with lines and rectangles')
ax[2].imshow(roi2_rgb)
ax[2].axis('off')
ax[2].set_title('ROI with lines and rectangles')
ax[3].imshow(cv2.cvtColor(thresholded, cv2.COLOR_BGR2RGB))
ax[3].axis('off')
ax[3].set_title('Thresholded Image')
#cv2.imwrite('image.jpg', image)
#cv2.imwrite('roi.jpg', roi)


plt.show()
