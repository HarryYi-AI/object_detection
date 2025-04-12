import json
import torch
import torchvision.transforms as transforms
import torchvision.models.detection as detection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import cv2
from PIL import Image
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.efficientnet_pytorch import EfficientNet

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# Load EfficientNet model
model = EfficientNet.from_pretrained('efficientnet-b7')
model = model.to(device)  # 将模型移动到GPU

# Load ImageNet class names
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'labels_map.txt')
labels_map = json.load(open(config_path))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Load object detection model with new weights parameter
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
obj_detection_model = detection.fasterrcnn_resnet50_fpn(weights=weights)
obj_detection_model = obj_detection_model.to(device)  # 将模型移动到GPU
obj_detection_model.eval()

# Preprocess image
tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

# Video capture
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sample_video.mp4')
cap = cv2.VideoCapture(data_path)

# Set frame skip factor
frame_skip = 5  # Process every 5th frame

frame_count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Skip frames if needed
    if frame_count % frame_skip != 0:
        continue

    # Convert frame to PIL image
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = tfms(pil_img).unsqueeze(0)
    img = img.to(device)  # 将图像数据移动到GPU

    # Object detection
    with torch.no_grad():
        boxes = obj_detection_model(img)[0]['boxes']
        scores = obj_detection_model(img)[0]['scores']

    # Filter boxes based on confidence threshold
    threshold = 0.5  # Adjust as needed
    filtered_boxes = [box for i, box in enumerate(boxes) if scores[i] > threshold]

    # Classify
    model.eval()
    with torch.no_grad():
        outputs = model(img)

    # Print predictions
    print('-----')
    for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))

    # Draw bounding boxes on the frame
    for box in filtered_boxes:
        box = [int(i) for i in box.tolist()]
        # Adjust box coordinates to fit within frame boundaries
        box[0] = max(0, box[0])
        box[1] = max(0, box[1])
        box[2] = min(frame.shape[1], box[2])
        box[3] = min(frame.shape[0], box[3])

        # Draw bounding box on the frame
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
