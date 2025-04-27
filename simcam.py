import cv2
import numpy as np
import time
import sys
import termios
import tty
import select

# CONFIG
USE_EDGE_DETECTION = True
#USE_EDGE_DETECTION = False
INVERT_PIXELS = False
target_width = 128
target_height = 32

# Initial crop height
virtual_crop_height = 128  # how much vertical window to grab
min_crop_height = 32      # minimum crop
max_crop_height = 480     # maximum crop (adjust if needed based on camera)

cap = cv2.VideoCapture(0)

# Framebuffer open
try:
    fb = open('/dev/fb0', 'wb')
    use_fb = True
except Exception as e:
    print(f"Warning: could not open /dev/fb0: {e}")
    use_fb = False

# Terminal setup
orig_settings = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin)

mirror_horizontal = False

def key_pressed():
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge detection
        if USE_EDGE_DETECTION:
            processed = cv2.Canny(gray, 50, 150)
        else:
            processed = gray

        # Frame size
        h, w = processed.shape

        # CROP full width, center vertically
        center_y = h // 2
        half_crop = virtual_crop_height // 2

        top = max(center_y - half_crop, 0)
        bottom = min(center_y + half_crop, h)

        cropped = processed[top:bottom, :]

        # Resize cropped area to 128x32
        small = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Mirror if needed
        if mirror_horizontal:
            small = cv2.flip(small, 1)

        # Threshold
        _, bw = cv2.threshold(small, 50, 255, cv2.THRESH_BINARY)
        bw = bw // 255

        if INVERT_PIXELS:
            bw = 1 - bw

        # ==== Terminal Display ====
        print("\033c", end="")
        print(f"Mirror: {mirror_horizontal} | Crop Height: {virtual_crop_height}")
        for y in range(target_height):
            line = ''
            for x in range(target_width):
                line += '#' if bw[y, x] else ' '
            print(line)

        # ==== Framebuffer Write ====
        if use_fb:
            packed = np.packbits(bw, axis=1)
            # Reverse bits inside each byte
        
            def reverse_bits(byte):
                return int('{:08b}'.format(byte)[::-1], 2)

            # Apply to all bytes
            packed_reversed = np.vectorize(reverse_bits)(packed).astype(np.uint8)

            if packed.shape == (target_height, target_width // 8):
                fb.seek(0)
                fb.write(packed_reversed.tobytes())
                fb.flush()
            else:
                print("Packing error!", packed.shape)

        # Handle keys
        if key_pressed():
            ch = sys.stdin.read(1)
            if ch == 'm':
                mirror_horizontal = not mirror_horizontal
                print(f">>> Mirror horizontal: {mirror_horizontal}")
            elif ch == 'e':
                USE_EDGE_DETECTION = not USE_EDGE_DETECTION
                print(f">>> Mirror horizontal: {mirror_horizontal}")

            elif ch == 'i':
                INVERT_PIXELS = not INVERT_PIXELS
                print(f">>> Mirror horizontal: {mirror_horizontal}")

            elif ch == '+':
                virtual_crop_height = min(virtual_crop_height + 4, max_crop_height)
                print(f">>> Crop height increased: {virtual_crop_height}")
            elif ch == '-':
                virtual_crop_height = max(virtual_crop_height - 4, min_crop_height)
                print(f">>> Crop height decreased: {virtual_crop_height}")
            elif ch == 'q':
                break
        #print("bw shape:", bw.shape, "bw min/max:", bw.min(), bw.max())
        #print("packed shape:", packed.shape, "packed size:", len(packed.tobytes()))
        time.sleep(0.001)

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)
    cap.release()
    if use_fb:
        fb.close()
    print("Exited cleanly.")

