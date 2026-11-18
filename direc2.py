import cv2
from ultralytics import YOLO

input_video_path = "./video2.mp4"
output_video_path = "./direc_arrows2.mp4"

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

# Define partition boundaries (35% left, 65% right)
left_boundary = int(width * 0.35)
right_boundary = int(width * 0.65)

print(f"Processing frames with professional directional arrow overlays...")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.4, verbose=False)
    annotated_frame = results[0].plot()
    
    # Track hazards per frame to decide the dominant guidance cue
    left_hazard = False
    right_hazard = False
    front_hazard = False
    
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        x_min, y_min, x_max, y_max = xyxy
        x_center = (x_min + x_max) / 2
        
        # Categorize spatial position
        if x_center < left_boundary:
            left_hazard = True
        elif x_center > right_boundary:
            right_hazard = True
        else:
            front_hazard = True

    # --- DRAW GRAPHIC DIRECTIONAL ARROWS & HUD OVERLAYS ---
    # Center HUD Banner Background
    cv2.rectangle(annotated_frame, (int(width*0.2), 20), (int(width*0.8), 80), (0, 0, 0), -1)
    cv2.rectangle(annotated_frame, (int(width*0.2), 20), (int(width*0.8), 80), (255, 255, 255), 2)

    # Priority 1: Front / Direct Hazard (Stairs / Blockage) -> STOP / UP ALERT
    if front_hazard:
        text = "STOP / HAZARD AHEAD"
        color = (0, 0, 255) # Red
        # Draw a warning upward/stop box marker
        cv2.putText(annotated_frame, text, (int(width*0.22), 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw a Front Warning Icon (Exclamation / Up Arrow pointing down to stop)
        cv2.arrowedLine(annotated_frame, (int(width/2), 140), (int(width/2), 90), (0, 0, 255), 6, tipLength=0.5)

    # Priority 2: Obstacle on Left -> Prompt user to move RIGHT (Right Arrow)
    elif left_hazard:
        text = "OBSTACLE LEFT -> MOVE RIGHT"
        color = (0, 165, 255) # Orange
        cv2.putText(annotated_frame, text, (int(width*0.21), 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw a big graphical RIGHT Arrow pointing ->
        cv2.arrowedLine(annotated_frame, (int(width*0.75), 130), (int(width*0.88), 130), (0, 165, 255), 6, tipLength=0.4)

    # Priority 3: Obstacle on Right -> Prompt user to move LEFT (Left Arrow)
    elif right_hazard:
        text = "OBSTACLE RIGHT -> MOVE LEFT"
        color = (0, 165, 255) # Orange
        cv2.putText(annotated_frame, text, (int(width*0.21), 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw a big graphical LEFT Arrow pointing <-
        cv2.arrowedLine(annotated_frame, (int(width*0.25), 130), (int(width*0.12), 130), (0, 165, 255), 6, tipLength=0.4)

    # Priority 4: Path Clear -> Safe Forward Arrow
    else:
        text = "PATH CLEAR -> PROCEED"
        color = (0, 255, 0) # Green
        cv2.putText(annotated_frame, text, (int(width*0.28), 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw a Forward Green Up Arrow ^
        cv2.arrowedLine(annotated_frame, (int(width/2), 140), (int(width/2), 90), (0, 255, 0), 6, tipLength=0.5)

    out.write(annotated_frame)
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Progress: Processed {frame_count}/{total_frames} frames...")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Success! Professional directional arrow simulation saved to {output_video_path}")