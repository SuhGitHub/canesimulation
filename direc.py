import cv2
from ultralytics import YOLO

input_video_path = "./video4.mp4"
output_video_path = "./direc4.mp4"

print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Could not open input video at {input_video_path}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Define partition boundaries based on frame width
left_boundary = int(width * 0.35)
right_boundary = int(width * 0.65)

print(f"Processing frames with directional spatial mapping...")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.4, verbose=False)
    annotated_frame = results[0].plot()
    
    # Draw spatial partition lines on the video for visual simulation
    cv2.line(annotated_frame, (left_boundary, 0), (left_boundary, height), (255, 255, 0), 2)
    cv2.line(annotated_frame, (right_boundary, 0), (right_boundary, height), (255, 255, 0), 2)
    
    # Evaluate bounding boxes for directional cues
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy() # [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = xyxy
        x_center = (x_min + x_max) / 2
        
        # Determine direction
        if x_center < left_boundary:
            direction_text = "Obstacle Left: Move Right"
            text_color = (0, 165, 255) # Orange
        elif x_center > right_boundary:
            direction_text = "Obstacle Right: Move Left"
            text_color = (0, 165, 255)
        else:
            direction_text = "STOP / Direct Hazard Ahead"
            text_color = (0, 0, 255) # Red
            
        # Overlay guidance text on frame
        cv2.putText(annotated_frame, direction_text, (int(x_min), max(30, int(y_min) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    out.write(annotated_frame)
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Progress: Processed {frame_count}/{total_frames} frames...")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Success! Directional simulation saved to {output_video_path}")