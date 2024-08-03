import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as T
import cv2

model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

def get_segmentation(image, model):
    with torch.no_grad():
        prediction = model([image])
    return prediction

image_path = "data/cups_initial.png"
image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image)

predictions = get_segmentation(image_tensor, model)
pred = predictions[0]

image_np = np.array(image)

for i in range(len(pred["boxes"])):
    if pred['scores'][i] > 0.3:  # You can set a threshold for confidence score
        box = pred["boxes"][i].numpy().astype(int)
        label = COCO_INSTANCE_CATEGORY_NAMES[pred['labels'][i]]
        score = pred['scores'][i].item()
        cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(image_np, f"{label}: {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        mask = pred["masks"][i, 0].numpy()
        mask = (mask > 0.5).astype(np.uint8)
        mask_rgb = np.zeros_like(image_np)
        mask_rgb[:, :, 0] = mask * 255  # Red channel for mask
        image_np = cv2.addWeighted(image_np, 1, mask_rgb, 0.5, 0)

plt.figure(figsize=(10, 10))
plt.imshow(image_np)
plt.axis('off')
plt.show()
