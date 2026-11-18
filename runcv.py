import cv2
from ultralytics import YOLO

# Path to the video generated from your SANPO frames
input_video_path = "./video4.mp4"
output_video_path = "./annot4.mp4"

# Load a lightweight YOLOv8 model (e.g., YOLOv8n for fast inference)
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")  # Will automatically download weights if not present

# Open input video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Could not open input video at {input_video_path}")
    exit()

# Get video properties for the writer
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Setup video writer for the annotated output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Processing {total_frames} frames with object detection...")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLOv8 inference on the current frame
    # conf=0.3 filters out low-confidence detections
    results = model(frame, conf=0.3, verbose=False)
    
    # Annotate the frame with bounding boxes and labels
    annotated_frame = results[0].plot()
    
    # Write the annotated frame to output video
    out.write(annotated_frame)
    
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Progress: Processed {frame_count}/{total_frames} frames...")

# Clean up resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Success! Annotated simulation video saved to {output_video_path}")