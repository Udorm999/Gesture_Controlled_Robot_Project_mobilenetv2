import cv2
import os

# ==========================
# CONFIG
# ==========================
INPUT_DIR = "/Users/vid/Desktop/gesture/dataset/open-palm/"
OUTPUT_DIR = "/Users/vid/Desktop/gesture/cropped_images/open-palm/"
FINAL_SIZE = 224
ZOOM_OUT_SIZE = 1080

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# ZOOM-OUT CENTER CROP
# ==========================
def zoom_out_center_crop(img, crop_size=ZOOM_OUT_SIZE, final_size=224):
    h, w, _ = img.shape

    # Resize if image too small
    if h < crop_size or w < crop_size:
        scale = crop_size / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w, _ = img.shape

    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2

    crop = img[start_y:start_y + crop_size,
               start_x:start_x + crop_size]

    # Resize back to model input size
    crop = cv2.resize(crop, (final_size, final_size))

    return crop

# ==========================
# PROCESS IMAGES
# ==========================
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Skipped: {filename}")
            continue

        cropped = zoom_out_center_crop(img)

        save_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(save_path, cropped)

        print(f"✅ Processed: {filename}")

print("🎉 Done! Zoomed-out center crops saved.")
