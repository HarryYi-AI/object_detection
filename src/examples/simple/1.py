# # # import cv2
# # # import numpy as np
# # # import tensorflow as tf
# # #
# # # # 加载预训练的 SSD-MobileNet 模型
# # # model = tf.saved_model.load("ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model")
# # #
# # # # 加载类别标签
# # # with open("coco_labels.txt", "r") as f:
# # #     labels = f.read().strip().split("\n")
# # #
# # # # 打开视频文件
# # # video_capture = cv2.VideoCapture("sample_video.mp4")
# # #
# # # # 初始化目标信息
# # # target_bbox = None
# # #
# # # while True:
# # #     ret, frame = video_capture.read()
# # #     if not ret:
# # #         break  # 如果没有读到帧，结束循环
# # #
# # #     # 图像预处理
# # #     processed_frame = cv2.resize(frame, (320, 320))
# # #     processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
# # #     processed_frame = processed_frame / 255.0
# # #     input_tensor = tf.convert_to_tensor(processed_frame, dtype=tf.float32)
# # #     input_tensor = input_tensor[tf.newaxis, ...]
# # #
# # #     # 进行目标检测
# # #     detections = model(input_tensor)
# # #
# # #     # 获取检测结果
# # #     boxes = detections["detection_boxes"][0].numpy()
# # #     scores = detections["detection_scores"][0].numpy()
# # #     classes = detections["detection_classes"][0].numpy().astype(int)
# # #
# # #     # 选择置信度最高的检测结果
# # #     if np.max(scores) > 0.5:
# # #         max_score_index = np.argmax(scores)
# # #         target_bbox = boxes[max_score_index]
# # #
# # #     # 在帧上绘制目标框
# # #     if target_bbox is not None:
# # #         h, w, _ = frame.shape
# # #         ymin, xmin, ymax, xmax = target_bbox
# # #         xmin = int(xmin * w)
# # #         xmax = int(xmax * w)
# # #         ymin = int(ymin * h)
# # #         ymax = int(ymax * h)
# # #
# # #         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
# # #         cv2.putText(frame, f"{labels[classes[max_score_index]]}: {scores[max_score_index]:.2f}",
# # #                     (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# # #
# # #     # 显示处理后的帧
# # #     cv2.imshow('Video', frame)
# # #
# # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # #         break
# # #
# # # # 释放视频流和窗口
# # # video_capture.release()
# # # cv2.destroyAllWindows()
# # #
#
#
# import json
# from PIL import Image
# import torch
# from torchvision import transforms
#
# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b7')
#
# # Preprocess image
# tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
# img = tfms(Image.open('img.jpg')).unsqueeze(0)
# print(img.shape) # torch.Size([1, 3, 224, 224])
#
# # Load ImageNet class names
# labels_map = json.load(open('labels_map.txt'))
# labels_map = [labels_map[str(i)] for i in range(1000)]
#
# # Classify
# model.eval()
# with torch.no_grad():
#     outputs = model(img)
#
# # Print predictions
# print('-----')
# for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
#     prob = torch.softmax(outputs, dim=1)[0, idx].item()
#     print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))


import json
from PIL import Image
import torch
from torchvision import transforms
import cv2
import numpy as np
from efficientnet_pytorch import EfficientNet

# Load pre-trained EfficientNet model
model = EfficientNet.from_pretrained('efficientnet-b7')

# Preprocess image
tfms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load ImageNet class names
labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Load YOLOv3 for object detection
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Open video capture (you can replace "input_video.mp4" with your video file)
cap = cv2.VideoCapture("sample_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        label = str(class_ids[i])
        confidence = confidences[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{labels_map[int(label)]} {confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Video recognition
    img = tfms(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(img)

    # Print predictions
    print('-----')
    for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob * 100))

    # Display the frame
    cv2.imshow('Video Recognition and Object Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
