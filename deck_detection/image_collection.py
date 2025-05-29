import requests
import cv2
import numpy as np
import os
import time

# ---- CONFIGURE THIS ----
url = "http://10.63.5.42:5000/video_feed2"  # <-- Your stream URL
output_dir = "images"
start_count = 60
total_images_to_collect = 120
# -------------------------

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open HTTP connection
    stream = requests.get(url, stream=True)
    if stream.status_code != 200:
        print(f"Failed to connect, status code {stream.status_code}")
        return

    bytes_buffer = b''
    count = 60

    for chunk in stream.iter_content(chunk_size=1024):
        bytes_buffer += chunk
        a = bytes_buffer.find(b'\xff\xd8')  # JPEG start
        b = bytes_buffer.find(b'\xff\xd9')  # JPEG end
        if a != -1 and b != -1:
            jpg = bytes_buffer[a:b+2]  # Extract full JPG
            bytes_buffer = bytes_buffer[b+2:]  # Cut processed part
           
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if frame is not None:
                filename = os.path.join(output_dir, f"frame_{count:04d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")
                count += 1
                time.sleep(0.5)  # Add small delay if you want slower capture

            if count >= total_images_to_collect:
                break

    print(f"Finished collecting {count} images.")

if __name__ == "__main__":
    main()
