# Patch classification

DigiLut is an object detection challenge. Yet the images are too big to apply YOLO/DETR models on them directly. Thus, I decided to rephrase the problem as a [**patch classification problem**](https://paperswithcode.com/task/image-classification), followed by a [**post processing**](./postprocessing.md) step, that converts predicted patches heat maps into bounding boxes.

**Steps**:

1. Convert the dataset of WSI slides into a labelled dataset of JPG patches
2. Train a patch classification classifier

!!! quote ""
    ``` mermaid
    graph TD
      A[Whole Slide Images dataset] -->|PyFAST patchifies and keeps patches with tissue in it| C[Patches 256x256];
      C --> D[Assign binary labels to patches];
      D --> H[Train/Test split over patient IDs];
      H --> E[Train patches];
      H --> F[Test patches];
      E --> K[Undersample the train dataset to ensure a better label balance]
      K --> L[Cross validation split: GroupKFold over the patient IDs]
      L --> N[Train a ResNet model]
      N -->|Repeat over N folds| L;
      N --> O[Monitor training with Tensorboard]
      N ---> Q[N models trained = 1 ensemble of models];
      N --> P[Log experiment configuration and metrics in MLflow]
      F --------> R[Predict on test samples with the ensemble of models];
      Q --> R;
    ```

---

## Dataset creation

- Whole Slide Images (WSI .tiff images) are too big, we tile them into $256 \times 256$ patches, at $\times 20$ magnification level.
- FAST, a medical image processing library, is used to tile the images into patches. FAST has tissue segmentation tools, so we only save patches that contain cells. This saves a lot of memory and I/O operations because most of the WSI is white background.
- Then we assign binary (0/1) labels to each patch:
    - 1 (positive) if IoU > threshold, with one of the ground truth bounding boxes of the slide
    - 0 (negative) else.

Doing so we save around 10k patches per slide. The labels are highly imbalanced (1:1000 positive)

## Training

- We run cross validation (n=5) using a groupKFold strategy over the patient IDs to avoid train/test leakage.
- We then train a simple MLP classifier with a pretrained ResNet backbone using a BCE or focal loss
    - I considered other models pretrained on WSI images (like Phikon from Owkin), but they were too big for my machine. Maybe a LORA approach could have mitigated this issue.
- By infering with this classifier on the whole slide, we get a scatter point of positive and negative patches at (x,y) coordinates.
