from turn_classifier import TurnClassifier
import torch
import cv2
from typing import Tuple
import numpy as np

def viz_mg_data(data: Tuple[torch.Tensor, int], model: TurnClassifier):
    images, label = data
    output = model(images.unsqueeze(0))

    for i in range(images.shape[0]):
        image = images[i].permute(1, 2, 0).detach().cpu().numpy()
        image = cv2.resize(image, (448, 448))
    
        print("label: ", label)
        print("raw model output: ", torch.nn.functional.softmax(output, dim=0))
        # printing percentage of turning
        label_str = "label = turning" if torch.argmax(output[0]) == 1 else "label = not turning"
        percent_str = f"turn pred: {torch.nn.functional.softmax(output[0])[1].item() * 100:.2f}%"
        print(label_str)
        print(percent_str)
        cv2.putText(image, label_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, percent_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey(0)

def viz_live_data(model: TurnClassifier):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (448, 448))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
        frame = frame.unsqueeze(0)
        output = model(frame)

        label_str = "label = turning" if torch.argmax(output[0]) == 1 else "label = not turning"
        percent_str = f"turn pred: {torch.nn.functional.softmax(output[0])[1].item() * 100:.2f}%"
        cv2.putText(frame, label_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, percent_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("frame", frame.permute(1, 2, 0).detach().cpu().numpy())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def center_crop(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = (w - size[0]) // 2
    y1 = (h - size[1]) // 2
    x2 = x1 + size[0]
    y2 = y1 + size[1]
    return img[y1:y2, x1:x2]
