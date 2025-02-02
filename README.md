# -Extraction-of-Cervical-Spine-Radiographic-Predictions-for-Kyphotic-Deformity-Following-Laminoplasty
Autonomous Extraction of Cervical Spine Radiographic Predictions for Kyphotic Deformity Following Laminoplasty


> Specs and Software used for this training
```
Thu Jan 30 10:33:30 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX A4500               On  |   00000000:C2:00.0 Off |                  Off |
| 30%   26C    P8             17W /  200W |       2MiB /  20470MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```
> Current set variables
`batch size` is 20
`number of class` is 1
`seed` is 17

> Refer to cell 16 to check if your GPU is availiable and information about the architecture of AI model
> Cell 15 contains code about the actual architecture model

> These are custom made Loss and Metrices function 
```
import tensorflow as tf
from tensorflow.keras import backend as K

def tversky_loss(y_true, y_pred, alpha=0.23, beta=0.99, smooth=1):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    true_pos = K.sum(y_true * y_pred)
    false_neg = K.sum(y_true * (1 - y_pred))
    false_pos = K.sum((1 - y_true) * y_pred)
    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
    return 1 - tversky_index
```
```
#METRICS
# IoU Score Metrics
def iou_score(y_true, y_pred):
    smoothing_factor = 1
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    iou = K.mean((intersection + smoothing_factor) / (union + smoothing_factor), axis=0)
    return iou

def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    
    intersection = K.sum(flat_y_true * flat_y_pred)
    union = K.sum(flat_y_true) + K.sum(flat_y_pred)
    
    dice = (2. * intersection + smoothing_factor) / (union + smoothing_factor)
    return dice
```


