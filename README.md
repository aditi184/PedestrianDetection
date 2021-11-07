# Pedestrian Detection

`PennFudanPed_train.json`:  Contains COCO annotations for a randomly generated train split of the PennFudan dataset. 

`PennFudanPed_val.json`:  Contains COCO annotations for the corresponding validation split of the PennFudan dataset. 

The below scripts should be run for detections obtained using all the three methods mentioned below:

1. Pretrained HoG
2. Custom HoG trained using SVM on HoG features
3. Pretrained Faster RCNN 

## Installation
```bash
git clone https://github.com/sm354/Pedestrian-Detection.git
cd Pedestrian-Detection
pip install -r requirements.txt
```

##### Download Penn-Fudan Dataset

```
wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
```

## Running Models
#### 1. Pretrained HoG Detector

```bash
python eval_hog_pretrained.py --root <path to dataset root directory> --test <path to test json> --out <path to output json>
```

#### 2. Custom HoG Detector

**Training**

```bash
python train_hog_custom.py --root <path to dataset root directory> --train <path to train json> --model <path to save trained SVM model>
```

**Testing**

```bash
python eval_hog_custom.py --root <path to dataset root directory> --test <path to test json> --out <path to output json> --model <path to trained SVM model>
```

#### 3. Faster RCNN

```bash
python eval_faster_rcnn.py --root <path to dataset root directory> --test <path to test json> --out <path to output json> --model <path to pretrained Faster RCNN weights file>
```

### Evaluation script

    python eval_detections.py --gt <path to ground truth annotations json> --pred <path to detections json>

The script `eval_detections.py` takes in ground truth annotations and predicted detections for the evaluation dataset and computes the following metrics:

1. Average Precision, computed over 10 IOU thresholds in the range 0.5:0.05:0.95
2. Average Recall computed at 1 detection per image.
3. Average Recall comptued at 10 detections per image.

## Authors

- [Shubham Mittal](https://www.linkedin.com/in/shubham-mittal-6a8644165/)
- [Aditi Khandelwal](https://www.linkedin.com/in/aditi-khandelwal-991b1b19b/)

