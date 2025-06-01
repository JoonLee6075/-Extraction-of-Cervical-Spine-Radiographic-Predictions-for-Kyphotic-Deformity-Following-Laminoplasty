# -Extraction-of-Cervical-Spine-Radiographic-Predictions-for-Kyphotic-Deformity-Following-Laminoplasty
Autonomous Extraction of Cervical Spine Radiographic Predictions for Kyphotic Deformity Following Laminoplasty
> Program written in python by Samuel D. Pettersson, Natalia Anna Koc, Joon Lee

**Link to the Keras Model used in this project**
https://drive.google.com/file/d/1bkv26SXE4l4n2KFHinSQy4f9f7w7FcW3/view?usp=sharing

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
### 2. Set Up a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Required Packages
```bash
pip install tensorflow notebook numpy pandas matplotlib SimpleITK keras-preprocessing scikit-learn 
```
### 4. Download Model
Download the Keras Model from the link above.

### 5. Prepare DataSet
The model is designed to analyze sagittal X-ray scans where the C7 vertebra up to the external acoustic meatus is clearly visible. To ensure accurate results, it's important that the patient's lead gown is not positioned too high on the shoulders, as this can block the view of the C7 vertebral body.

Make sure to insert your datas inside /workspace/myxraydata/*.nrrd. If not you must edit the ipynb file to relocate the data directory.

### 6. Compile and Test
Now you need to load your data into preprocessing. You can use the same preprocessing method inside the ipynb file. Once it is done, 
```python
from tensorflow.keras.models import load_model

# Load your saved Keras model
model = load_model("model.weights.h5", compile=False)
```
Once Loaded, you may proceed to test out the prediction.
```python
prediction = model.predict(image)
...
```

https://github.com/JoonLee6075/-Extraction-of-Cervical-Spine-Radiographic-Predictions-for-Kyphotic-Deformity-Following-Laminoplasty.git
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


