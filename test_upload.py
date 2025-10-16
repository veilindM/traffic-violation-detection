# test_upload.py
from firebase_utils import upload_violation_image

# create a tiny test image file
from PIL import Image
img_path = "test_upload.jpg"
Image.new("RGB", (100,100), color=(255,0,0)).save(img_path)

url = upload_violation_image(img_path, "TESTPLATE", 1)
print("Uploaded URL:", url)
