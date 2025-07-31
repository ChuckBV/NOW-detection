# NOW-detection

ML workflow to detect NOW on sticky traps using scinet and yolo.

### Create and activate virtual environment
```bash
python3 -m venv venv_yolo
source venv_yolo/bin/activate
pip install -r requirements.txt
```


### Optional: Add virtual environment to Jupyter
```bash
pip install ipykernel
python -m ipykernel install --user --name=venv_yolo --display-name "YOLO (venv_yolo)"
```

### Example `your_data.yaml`
```yaml
train: path/to/train/images
val: path/to/val/images
names:
  - class1
  - class2
```

## Training Example
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="your_data.yaml",
    epochs=25,
    imgsz=640,
    batch=8,
    project="my_project",
    name="my_experiment"
)
```

## Key Scripts

### Data Prep and Processing
- `organise.py`: Arranges dataset into folders for traning 
- `rotate.py`: Rotates images 
- `fix.py`: Fixes broken annotations or paths
- `format_and_separate.py`: Moves and formats files by type
- `duplicate.py`: Detects and filters out duplicate files
- `voc.py`: Converts annotations to/from Pascal VOC format
- `rm-file.txt`: List of files to delete from dataset not needed for most applications


### Training and Evaluation
- `train.py`: Main training runner
- `train_now_6_25_2025.py`: Specific experiment training script
- `modelTest.py`: Runs predictions and outputs images with bounding boxes
- `model_analytics_demo.py`: Loads trained model and shows visual metrics
- `model_analytics_function.py`: Helper functions for plotting precision, recall, confidence, and loss


### Prediction and Metrics
- `demo.py`: Fast sample inference runner
- `accurate.py`: Compares predictions against ground truth
- `bet.py`: May experiment with detection thresholds or model output settings
- `count.py`: Tallies number of detections per image or set


## Config and Jobs
- `data-template.yml`: Template YOLO data config
- `job.sh`: SLURM shell script to run training on a cluster not needed for most
- `make-venv`: Bash script to set up Python virtual environment
- `setup.txt`: Extra setup notes and instructions
