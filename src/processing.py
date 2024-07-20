import time

import cv2
import torch
from PIL import Image
from ultralytics import YOLO

from training_classification import ResNetTrafficSignClassifier, transform

CONFIDENCE_THRESHOLD = 0.8  # Confidence threshold for sign detection


def initialize_models(device, detection_model_path, classification_model_path, num_classes):
    detection_model = YOLO(detection_model_path).to(device)
    classification_model = ResNetTrafficSignClassifier(num_classes).to(device)
    classification_model.load_state_dict(torch.load(classification_model_path))
    return detection_model, classification_model


def predict_bbox(frame, model, device, x_scale, y_scale):
    results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, device=device, verbose=False)
    results_list = []
    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()

            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)

            results_list.append((x1, y1, x2, y2, confidence))
    return results_list


def predict_class(roi_frame, model, device):
    roi_frame = Image.fromarray(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))
    roi_frame = transform(roi_frame).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(roi_frame)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_confidences, top_pred = torch.max(probabilities, 1)

    return top_pred[0].item(), top_confidences[0].item()


def inference(frame, inference_frame, x_scale, y_scale, class_names, detection_model, classification_model, device,
              debug_info):
    class_id, class_conf = -1, 0.0
    result_list = predict_bbox(inference_frame, detection_model, device, x_scale, y_scale)

    for result in result_list:
        x1, y1, x2, y2, bbox_conf = result
        roi_frame = frame[y1:y2, x1:x2]
        class_id, class_conf = predict_class(roi_frame, classification_model, device)
        if debug_info:
            frame = show_inference(frame, x1, y1, x2, y2, bbox_conf, class_names[class_id], class_conf)

    return frame, class_id, class_conf


def show_fps(frame, prev_frame_time):
    curr_frame_time = time.time()
    fps = 1 / (curr_frame_time - prev_frame_time)
    text = f"FPS: {fps:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (0, 0), (15 + text_width, 35 + text_height), (0, 0, 255), -1)
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    return frame, curr_frame_time


def show_sign(frame, sign_image, original_width, original_height):
    sign_image = cv2.resize(sign_image, (sign_image.shape[1] * 2, sign_image.shape[0] * 2))
    frame[original_height-sign_image.shape[0]:original_height, original_width-sign_image.shape[1]:original_width] \
        = sign_image
    return frame


def show_inference(frame, x1, y1, x2, y2, bbox_conf, class_name, class_conf):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
    text = f"{bbox_conf:.2f}, {class_name}: {class_conf:.2f}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
    cv2.rectangle(frame, (x1, y1 - 40), (x1 + text_size[0], y1 - 20 + text_size[1]), (0, 0, 255), -1)
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    return frame
