# Traditional Outfit Recognition Using Deep Learning

This repository presents a two-stage deep learning framework for automatic traditional clothing recognition using object detection and multi-output attribute classification.

The system first localizes garment regions using YOLOv8 and then predicts multiple fine-grained attributes from cropped regions using CNN and Transformer backbones.

---

## Project Overview

Traditional clothing recognition is challenging due to high visual similarity between garment categories and variations in sleeve structure and clothing type.

This project proposes a two-stage pipeline:

1. Garment detection using YOLOv8
2. Multi-output attribute classification using ROI crops

The classifier predicts:

- Outfit name (Qipao, Tangzhuang, Karabela)
- Clothing type (Dress, Shirt, Jacket)
- Sleeve type (Sleeveless, Short Sleeves, Long Sleeves, 3/4 Sleeves)

ROI-based classification improves fine-grained attribute prediction performance compared to full-image classification.

---

## Pipeline Architecture

Image  
→ YOLOv8 detection  
→ ROI cropping  
→ Multi-output classifier  
→ Attribute predictions

Final best system:

YOLOv8 + Ensemble (ResNet-18 + EfficientNet-B0 + DeiT-Small)

---

## Dataset

Custom curated dataset:

- **2,348 images**
- 3 traditional outfit classes:
  - Qipao (733 images)
  - Tangzhuang (509 images)
  - Karabela (1106 images)

Annotations:

- bounding boxes created using Roboflow
- attribute labels stored in CSV format

Attributes predicted:

- outfit name
- clothing type
- sleeve type

Dataset split into:

- training
- validation
- testing

Two classification strategies evaluated:

1. Full-image classification
2. ROI-based classification

ROI classification produced significantly better results.

---

## Detection Performance

Comparison between YOLOv8 and YOLOv11s:

| Model | Precision | Recall | mAP@50 | mAP@50–95 |
|------|-----------|--------|--------|-----------|
| YOLOv8 | 0.946 | 0.982 | **0.985** | **0.875** |
| YOLOv11s | **0.947** | 0.980 | 0.967 | 0.866 |

YOLOv8 achieved stronger overall localization performance and was selected for ROI extraction in the final pipeline.

---

## Multi-Output Classification Architecture

The classification network uses:

- shared backbone
- three prediction heads:
  - outfit name
  - clothing type
  - sleeve type

Backbones evaluated:

- ResNet-18
- EfficientNet-B0
- VGG16
- DeiT-Small (Vision Transformer)

Best-performing single backbone:

EfficientNet-B0

Overall attribute accuracy:

**0.9603**

Sleeve-type classification remains the most challenging prediction task.

---

## Ensemble Model Performance

Equal-voting ensembles were evaluated to improve attribute-level prediction accuracy.

Best-performing ensemble:

ResNet-18 + EfficientNet-B0 + DeiT-Small

ROI classification performance:

Accuracy: **0.9635**  
Precision: **0.9604**  
Recall: **0.9358**  
F1-score: **0.9470**

This hybrid CNN + Transformer ensemble achieved the best overall classification performance.

---

## One-Stage vs Two-Stage Pipeline Comparison

Performance comparison using identical ensemble classifiers:

| Pipeline | Accuracy | Precision | Recall | F1-score |
|---------|----------|-----------|--------|---------|
| Full-image classification | 0.9421 | 0.9038 | 0.8793 | 0.8894 |
| YOLOv8 + ROI classification | **0.9635** | **0.9604** | **0.9358** | **0.9470** |

Results confirm that ROI localization significantly improves fine-grained attribute prediction, particularly sleeve-type recognition.

---

## Repository Structure

dataset_loader.py  

models/
- multioutput_resnet.py
- multioutput_efficientnet.py
- multioutput_deit.py
- multioutput_vgg16.py

training scripts:

- train_multioutput_full_generic.py
- train_efficientnetb0_full_weighted.py

evaluation scripts:

- evaluate_multioutput_single_v4.py
- ensemble_multioutput_roi_v4.py

---

## Key Contributions

This work demonstrates:

- a two-stage detection-classification framework for traditional clothing recognition
- a multi-output attribute prediction architecture
- improved sleeve-type recognition using ROI localization
- hybrid CNN + Transformer ensemble learning
- comparison between one-stage and two-stage pipelines

---

## Applications

Potential applications include:

- cultural heritage digitization
- museum archive indexing
- intelligent fashion retrieval systems
- dataset annotation automation
- AI-assisted historical garment analysis

---

## Author

Stanley Alex Edgard Sanon  
M.S. Artificial Intelligence  
Tamkang University, Taiwan

Thesis:

Automated Traditional Outfit Recognition Based on Deep Learning
