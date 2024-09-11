from grounded import grounded
from robust_sam import robust_sam

image_path = "download.webp"
detections = grounded(image_path, "jacket")
bbox = detections.xyxy[0]

mask = robust_sam(image_path, bbox)