from turn_classifier import TurnClassifier
import torch
import cv2
from typing import Tuple

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