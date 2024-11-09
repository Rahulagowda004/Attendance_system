import cv2
import os

# Set up directory to save captured images
output_dir = "captured_images"
os.makedirs(output_dir, exist_ok=True)

# Number of images to capture
num_images = 100
capture_interval = 5  # Number of frames to skip between captures (adjust as needed)

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, change if multiple cameras are present

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to stop capturing early.")

count = 0
frame_count = 0
while count < num_images:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Capture images at intervals to avoid taking too many in quick succession
    if frame_count % capture_interval == 0:
        image_path = os.path.join(output_dir, f"image_{count:03}.png")
        cv2.imwrite(image_path, frame)
        print(f"Captured image {count + 1}/{num_images}")
        count += 1

    frame_count += 1

    # Display the captured frame (optional)
    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to stop early
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} images in '{output_dir}' directory.")
