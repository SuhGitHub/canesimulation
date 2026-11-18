import cv2
import os
import glob

# Adjust this to the exact folder containing your .png frames
frames_dir = "./sanpo_data/video_frames"
output_video_path = "./video4.mp4"

# Fetch and numerically sort all PNG files
image_files = sorted(
    glob.glob(os.path.join(frames_dir, "*.png")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) if os.path.splitext(os.path.basename(x))[0].isdigit() else x
)

if not image_files:
    print(f"Error: No PNG frames found in {frames_dir}. Check your path!")
    exit()

# Read the first frame to get dimensions
first_frame = cv2.imread(image_files[0])
if first_frame is None:
    print(f"Error: Could not read the first frame: {image_files[0]}")
    exit()

height, width, _ = first_frame.shape

# Setup video writer (using mp4v codec for standard .mp4 container)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30.0  # SANPO egocentric streams typically run at 30 fps
video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Stitching {len(image_files)} frames into {output_video_path} at {fps} FPS...")

for i, file_path in enumerate(image_files):
    frame = cv2.imread(file_path)
    if frame is None:
        print(f"Warning: Skipping unreadable frame: {file_path}")
        continue
    video.write(frame)
    
    if (i + 1) % 200 == 0 or (i + 1) == len(image_files):
        print(f"Progress: {i + 1}/{len(image_files)} frames processed...")

video.release()
cv2.destroyAllWindows()
print(f"Success! Video saved to {output_video_path}")