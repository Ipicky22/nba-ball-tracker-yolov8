# Learning Model Creation with an NBA Ball Tracker

![NBA Ball Tracker Logo](assets/logo.png)

---

## 📌 Objective

To build a robust dataset for training object detection and tracking models (YOLOv8), specifically designed to identify and follow basketballs in real NBA match conditions.

---

# Table of Contents

-   [Objective](#objective)
-   [1. Define the Dataset Source](#1-define-the-dataset-source)
-   [2. Create the Dataset](#2-create-the-dataset)
-   [3. Clean the Dataset](#3-clean-the-dataset)
    -   [3.1 Blur Filter](#31-blur-filter)
    -   [3.2 Light Filter](#32-light-filter)
    -   [3.3 Empty Frame Filter](#33-empty-frame-filter)
    -   [3.4 Duplicate Frame Filter](#34-duplicate-frame-filter)
    -   [3.5 Post-Cleaning Summary](#35-post-cleaning-summary)
-   [4. Building Our First Classification Model](#4-building-our-first-classification-model)
    -   [4.1 What is a Classification Model?](#41-what-is-a-classification-model)
    -   [4.2 Implementation](#42-implementation)
    -   [4.3 Results](#43-resultats)
    -   [4.4 Application on Our Dataset](#44-application-on-our-dataset)
-   [5. YOLO's Explanation](#5-yolos-explanation)
    -   [5.1 What is YOLO?](#what-is-yolo)
    -   [5.2 How Does YOLO Work?](#how-does-yolo-work)
    -   [5.3 Why YOLO?](#why-yolo)
    -   [5.4 YOLOv8 Model Variants](#yolov8-model-variants)
-   [6. Epochs Explanation](#6-epochs-explanation)
    -   [6.1 What is an Epoch?](#what-is-an-epoch)
    -   [6.2 Related Concepts](#related-concepts)
    -   [6.3 What to Monitor During Training](#what-to-monitor-during-training)
    -   [6.4 Why Multiple Epochs Are Needed](#why-multiple-epochs-are-needed)
-   [7. Building Our First Detection Model](#7-building-our-first-detection-model)
    -   [7.1 What is a Detection Model?](#71-what-is-a-detection-model)
    -   [7.2 What is an Annotation?](#72-what-is-an-annotation)
    -   [7.3 Implementation](#73-implementation)
    -   [7.4 Results](#74-resultats)
    -   [7.5 Application on Our Dataset](#75-application-on-our-dataset)
-   [8. And What's Next?](#and-whats-next)
    -   [Model on 4256 Images](#model-on-4256-images)
    -   [Model on 9310 Images](#model-on-9310-images)

---

## 1. Define the Dataset Source

-   **Source**: A 1-hour highlight compilation of the 2024–25 NBA season  
    [Watch on YouTube](https://www.youtube.com/watch?v=28shPp78KsE)
-   **Why this video?**
    -   Varied camera angles and lighting
    -   Multiple teams and jerseys
    -   Natural ball movements (dribbles, passes, rebounds, shots)

Using a highlight reel provides image diversity, reducing the risk of model bias (e.g., associating the ball with a specific player or team).

---

## 2. Create the Dataset

The full video is divided into twelve 5-minute clips and one final clip of less than 5 minutes using [`splitVideoToClip.py`](scripts/dataset/splitVideoToClip.py).  
This results in a `clips/` folder containing 13 short videos.

From each clip, we extract all the individual frames using [`splitClipToFrame.py`](scripts/dataset/splitClipToFrame.py).  
This results in a `batches/` folder containing 13 subfolders (batch_00 to batch_12).  
Each `batch` has an `images/` folder containing the extracted frames.

📦 Dataset overview:

| Batch     | Total  |
| --------- | ------ |
| batch_00  | 9167   |
| batch_01  | 8836   |
| batch_02  | 9106   |
| batch_03  | 8935   |
| batch_04  | 8930   |
| batch_05  | 8977   |
| batch_06  | 9097   |
| batch_07  | 8920   |
| batch_08  | 8983   |
| batch_09  | 9001   |
| batch_10  | 9076   |
| batch_11  | 8933   |
| batch_12  | 810    |
| **Total** | 108771 |

---

## 3. Clean the Dataset

Cleaning ensures better learning and generalization.  
We apply the following cleaning steps:

| Step                  | Description                                             | Script                                                      |
| --------------------- | ------------------------------------------------------- | ----------------------------------------------------------- |
| Blur filtering        | Removes blurry images (Laplacian variance < `150`)      | [`blurFilter.py`](scripts/dataset/blurFilter.py)            |
| Light filtering       | Removes overly dark or bright images                    | [`lightFilter.py`](scripts/dataset/lightFilter.py)          |
| Empty frame filtering | Removes black/empty/corrupted images (variance < `5.0`) | [`emptyFilter.py`](scripts/dataset/emptyFilter.py)          |
| Duplicate detection   | Removes near-identical images (SSIM > `0.89`)           | [`duplicateFilter.py`](scripts/dataset/duplicateFilter.py)` |

---

### 3.1 Blur Filter

Blurry images degrade training:

-   Objects are hard to identify even for humans
-   They introduce noise and lower model precision
-   They mislead the model in learning shapes and textures

**Metric**: Laplacian variance < `150`  
📜 Script: [`blurFilter.py`](scripts/dataset/blurFilter.py)

---

### 3.2 Light Filter

Too dark or too bright images:

-   Hide object details
-   Cause false detections or no detection
-   Hurt model generalization

**Thresholds**:

-   `dark_threshold = 20`
-   `bright_threshold = 230`

📜 Script: [`lightFilter.py`](scripts/dataset/lightFilter.py)

---

### 3.3 Empty Frame Filter

Empty or black frames:

-   Contain no useful visual data
-   Bias the dataset toward non-objects
-   Are often artifacts from video transitions

**Threshold**: pixel variance < `5.0`  
📜 Script: [`emptyFilter.py`](scripts/dataset/emptyFilter.py)

---

### 3.4 Duplicate Frame Filter

Duplicate frames:

-   Cause overfitting (same data over and over)
-   Reduce dataset diversity
-   Bias metrics if they appear in validation

**Similarity metric**: SSIM > `0.89` (compared to next 3 frames)  
📜 Script: [`duplicateFilter.py`](scripts/dataset/duplicateFilter.py)

---

### 3.5 Post-Cleaning Summary

After all filters, the dataset looks like this:

| Batch     | Usable         | Blur           | Dark      | Bright    | Duplicates   | Total  |
| --------- | -------------- | -------------- | --------- | --------- | ------------ | ------ |
| batch_00  | 7251 (79.10%)  | 1199 (13.08%)  | 0 (0.00%) | 0 (0.00%) | 717 (7.82%)  | 9167   |
| batch_01  | 6616 (74.88%)  | 1498 (16.95%)  | 0 (0.00%) | 0 (0.00%) | 722 (8.17%)  | 8836   |
| batch_02  | 7173 (78.76%)  | 1010 (11.09%)  | 0 (0.00%) | 0 (0.00%) | 923 (10.14%) | 9106   |
| batch_03  | 7095 (79.42%)  | 1099 (12.30%)  | 0 (0.00%) | 0 (0.00%) | 741 (8.29%)  | 8935   |
| batch_04  | 7277 (81.50%)  | 1097 (12.28%)  | 0 (0.00%) | 0 (0.00%) | 556 (6.23%)  | 8930   |
| batch_05  | 7345 (81.84%)  | 1026 (11.43%)  | 0 (0.00%) | 0 (0.00%) | 606 (6.75%)  | 8977   |
| batch_06  | 7045 (77.43%)  | 1131 (12.43%)  | 0 (0.00%) | 0 (0.00%) | 921 (10.12%) | 9097   |
| batch_07  | 7003 (78.50%)  | 1266 (14.19%)  | 0 (0.00%) | 0 (0.00%) | 651 (7.30%)  | 8920   |
| batch_08  | 7566 (84.22%)  | 660 (7.35%)    | 2 (0.02%) | 0 (0.00%) | 755 (8.40%)  | 8983   |
| batch_09  | 7037 (78.20%)  | 1110 (12.33%)  | 0 (0.00%) | 0 (0.00%) | 854 (9.49%)  | 9001   |
| batch_10  | 6828 (75.24%)  | 1552 (17.10%)  | 0 (0.00%) | 0 (0.00%) | 696 (7.67%)  | 9076   |
| batch_11  | 6839 (76.54%)  | 1365 (15.28%)  | 2 (0.02%) | 1 (0.01%) | 725 (8.12%)  | 8933   |
| batch_12  | 672 (82.96%)   | 52 (6.42%)     | 0 (0.00%) | 0 (0.00%) | 86 (10.62%)  | 810    |
| **Total** | 85748 (78.83%) | 14065 (12.93%) | 4 (0.00%) | 1 (0.00%) | 8953 (8.23%) | 108771 |

💡 We lost 23,023 images during the cleaning process — a trade-off for dataset quality.

---

## 4. Building Our First Classification Model

Before training a detection model, we must **classify images as “ball” or “no ball”**.

Instead of manually sorting all 85k images, we start with a **small subset** (batch_00) to train a binary **classification model** that will help us sort the rest.

---

### 4.1 What is a Classification Model?

A classification model is an AI algorithm that assigns a **label or class** to an input.

#### Summary:

> It answers: **“Which category does this belong to?”**

#### Types of classification:

-   **Binary** (e.g. “ball” or “no ball”)
-   **Multiclass** (e.g. “ball”, “player”, “referee”)
-   **Multilabel** (multiple classes at once)

In our case, we want to train a **binary classifier**.

---

### 4.2 Implementation

1. We manually sort `batch_00` into two folders: `ball/` and `no_ball/`, using [`manualClassification.py`](scripts/classification/manualClassification.py).
2. We split those into `train/` and `val/` using [`splitTrainValFolder.py`](scripts/classification/splitTrainValFolder.py).

Folder structure:

dataset/
├── train/
│ ├── ball/
│ └── without_ball/
├── val/
│ ├── ball/
│ └── without_ball/

-   `train/`: used to **train** the model
-   `val/`: used to **validate** its performance

Let's start training

```bash
yolo classify train \
  model=yolov8n-cls.pt \
  data=/dataset \
  epochs=30 \
  imgsz=224 \
  batch=32 \
```

### 4.3 Resultats

[`Consult here`](models/classification/)

### Confusion Matrix (Raw Counts)

|                     | Predicted: Ball | Predicted: No Ball |
| ------------------- | --------------- | ------------------ |
| **Actual: Ball**    | **801**         | 37                 |
| **Actual: No Ball** | 51              | **562**            |

#### Interpretation:

-   **801 True Positives**: correctly predicted frames containing a ball.
-   **562 True Negatives**: correctly predicted frames without a ball.
-   **37 False Negatives**: missed ball instances.
-   **51 False Positives**: falsely predicted a ball in empty frames.

---

### Normalized Confusion Matrix

-   **94%** of `ball` images are correctly classified.
-   **94%** of `no_ball` images are also correctly classified.
-   The model shows **balanced performance** across both classes.

---

### Training Curves

#### Loss Curves

-   Both **training and validation loss** consistently decrease.
-   No signs of overfitting – learning is stable and effective.

#### Accuracy

-   **Top-1 accuracy** reaches over **93%** on the validation set.
-   **Top-5 accuracy** remains at **100%**, as expected in binary classification.

---

## Summary

The model performs **very well** on this binary classification task:

-   High and balanced precision and recall
-   Low error rate (under 6% for both FP and FN)
-   Steady learning behavior through training

### 4.4 Application on our dataset

We sort our photos based on whether they contain a ball or not using our new classification model, [`ballFilter.py`](scripts/dataset/ballFilter.py).

At this stage, we get this:

| Batch     | Usable (%)    | Blur (%)      | Dark (%)  | Bright (%) | Duplicates (%) | Without_ball (%) | Total      |
| --------- | ------------- | ------------- | --------- | ---------- | -------------- | ---------------- | ---------- |
| batch_00  | 46.43% (4256) | 13.08% (1199) | 0.00% (0) | 0.00% (0)  | 7.82% (717)    | 32.67% (2995)    | 9167       |
| batch_01  | 57.20% (5054) | 16.95% (1498) | 0.00% (0) | 0.00% (0)  | 8.17% (722)    | 17.68% (1562)    | 8836       |
| batch_02  | 58.66% (5342) | 11.09% (1010) | 0.00% (0) | 0.00% (0)  | 10.14% (923)   | 20.11% (1831)    | 9106       |
| batch_03  | 61.41% (5487) | 12.30% (1099) | 0.00% (0) | 0.00% (0)  | 8.29% (741)    | 18.00% (1608)    | 8935       |
| batch_04  | 62.81% (5609) | 12.28% (1097) | 0.00% (0) | 0.00% (0)  | 6.23% (556)    | 18.68% (1668)    | 8930       |
| batch_05  | 65.38% (5869) | 11.43% (1026) | 0.00% (0) | 0.00% (0)  | 6.75% (606)    | 16.44% (1476)    | 8977       |
| batch_06  | 56.46% (5136) | 12.43% (1131) | 0.00% (0) | 0.00% (0)  | 10.12% (921)   | 20.98% (1909)    | 9097       |
| batch_07  | 54.38% (4851) | 14.19% (1266) | 0.00% (0) | 0.00% (0)  | 7.30% (651)    | 24.13% (2152)    | 8920       |
| batch_08  | 62.74% (5636) | 7.35% (660)   | 0.02% (2) | 0.00% (0)  | 8.40% (755)    | 21.49% (1930)    | 8983       |
| batch_09  | 56.17% (5056) | 12.33% (1110) | 0.00% (0) | 0.00% (0)  | 9.49% (854)    | 22.01% (1981)    | 9001       |
| batch_10  | 61.46% (5578) | 17.10% (1552) | 0.00% (0) | 0.00% (0)  | 7.67% (696)    | 13.77% (1250)    | 9076       |
| batch_11  | 59.48% (5313) | 15.28% (1365) | 0.02% (2) | 0.01% (1)  | 8.12% (725)    | 17.09% (1527)    | 8933       |
| batch_12  | 66.54% (539)  | 6.42% (52)    | 0.00% (0) | 0.00% (0)  | 10.62% (86)    | 16.42% (133)     | 810        |
| **Total** | **63726**     | **14065**     | **4**     | **1**      | **8953**       | **22022**        | **108771** |

So we have 22,022 images without the ball, leaving 63,728 usable ones.

## 5. Yolo's Explanation

### What is YOLO?

**YOLO** (You Only Look Once) is a real-time object detection algorithm that can both locate and classify multiple objects in an image in a single forward pass of a neural network.

---

### Overview

YOLO is designed to:

-   Detect **where** objects are in an image (bounding boxes)
-   Identify **what** those objects are (classification)
-   Do this **extremely fast**, suitable for real-time applications

---

### How Does YOLO Work?

1. The input image is divided into a grid (e.g., 7×7).
2. Each grid cell predicts:
    - One or more **bounding boxes** (position and size)
    - A **confidence score** (is there an object?)
    - The **class label** (e.g., ball, person, car)
3. All predictions are combined and filtered to return final detections.

---

### Why YOLO?

| Feature         | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| Single pass     | The image is processed in a single forward pass of the model |
| Real-time speed | Ideal for applications requiring speed, like video or drones |
| Good accuracy   | Balances detection performance and inference time            |
| Versatile       | Works across various tasks and environments                  |

---

### YOLOv8: The Latest Version

YOLOv8, developed by **Ultralytics**, extends the core object detection capabilities and introduces:

-   Object detection
-   Image classification
-   Instance segmentation
-   Pose estimation

YOLOv8 is modular, flexible, and highly optimized for production use.

---

### YOLOv8 Model Variants

| Model   | Size    | Speed (FPS)\* | Accuracy (mAP50) | Parameters | Recommended Dataset Size |
| ------- | ------- | ------------- | ---------------- | ---------- | ------------------------ |
| YOLOv8n | Nano    | Very Fast     | Lower            | ~3.2M      | < 5,000 images           |
| YOLOv8s | Small   | Fast          | Good             | ~11.2M     | 5,000 – 15,000 images    |
| YOLOv8m | Medium  | Moderate      | High             | ~25.9M     | 10,000 – 50,000 images   |
| YOLOv8l | Large   | Slower        | Very High        | ~43.7M     | 50,000 – 200,000 images  |
| YOLOv8x | X-Large | Slowest       | Highest          | ~68.2M     | 100,000+ images          |

\* Approximate FPS on standard GPU – may vary by system

---

### Summary

YOLO is one of the most efficient and widely-used object detection algorithms.  
It processes images quickly and accurately, making it a top choice for both research and real-world deployment.

---

## 5. Epoch's Explanation

### What is an Epoch?

An **epoch** is one full pass through the entire training dataset by the learning algorithm.

During an epoch:

-   The model sees every example in the training set **once**
-   It performs forward passes, computes losses, and updates weights
-   The goal is to progressively improve the model's performance

Typically, models are trained over **multiple epochs** to allow learning to converge.

---

#### Related Concepts

-   **Batch**: A subset of the dataset used for one training step. If the dataset has 10,000 images and the batch size is 100, then each epoch consists of 100 steps.
-   **Step/Iteration**: One update of the model weights, using one batch of data.

---

### What to Monitor During Training

#### 1. Training Loss

-   Measures how well the model fits the training data
-   Should generally **decrease** over time
-   Common loss types: CrossEntropy, MSE, Focal Loss

#### 2. Validation Loss

-   Computed on the **validation dataset**, which is not seen during training
-   Helps monitor **generalization**
-   If it increases while training loss decreases, this may indicate **overfitting**

#### 3. Accuracy

-   Percentage of correct predictions
-   Can be computed for both training and validation sets
-   Often used in classification tasks

#### 4. mAP (Mean Average Precision)

-   Used in detection tasks (e.g., YOLO)
-   Measures how well the predicted boxes match ground truth
-   `mAP@0.5`: Intersection over Union (IoU) threshold of 0.5
-   `mAP@0.5:0.95`: Average over multiple IoU thresholds

#### 5. Precision and Recall

-   **Precision**: Out of predicted positives, how many were correct
-   **Recall**: Out of actual positives, how many were detected
-   High precision with low recall = few false positives
-   High recall with low precision = few false negatives

#### 6. Learning Rate

-   The step size used to update weights
-   Should be tuned carefully; too high can destabilize training, too low can slow it down

---

### Why Multiple Epochs Are Needed

-   A single epoch is often not enough for the model to learn well
-   More epochs allow the model to refine its weights
-   However, too many epochs can lead to **overfitting** if not regularized

To find the best number of epochs, use **early stopping** or monitor **validation loss** trends.

---

### Summary

An epoch represents one complete cycle through the training data.  
By analyzing training and validation loss, accuracy, and other metrics across epochs, we can understand how well a model is learning and generalizing.

---

## 6. Building Our First Detection Model

I annotated the first 1100 images of batch_00 using the CVat tool.
"Annotated" means that I circled the ball in these images.

Using these annotations, I'll create my first detection model, which will then annotate for me and save me time.

---

### 6.1 What is a Detection Model?

A **detection model** is designed to both **locate** and **identify** multiple objects in an image.

### Characteristics:

-   Input: One image
-   Output: One or more **bounding boxes**, each with a class label and confidence score
-   Capable of detecting **multiple objects**, even of different classes
-   Provides **spatial information** (where each object is)

---

### 6.2 Qu'est ce qu'une annotation ?

An **annotation** is metadata added to an image that tells a machine learning model what to learn.  
In object detection, an annotation describes **what object is in the image** and **where it is located**.

Annotations are typically stored in separate files (often `.txt`) that accompany the images.  
Each file contains information about the objects present in the corresponding image.

## Annotation Format (YOLO)

YOLO uses a specific `.txt` format where each **line** represents one object in the image.

### Line Structure:

<class_id> <x_center> <y_center> <width> <height>

All coordinates are **normalized** between 0 and 1 relative to the image size.

### Field Definitions:

| Field      | Description                                            |
| ---------- | ------------------------------------------------------ |
| `class_id` | Integer representing the object class (e.g., 0 = ball) |
| `x_center` | Horizontal center of the bounding box (normalized)     |
| `y_center` | Vertical center of the bounding box (normalized)       |
| `width`    | Width of the bounding box (normalized)                 |
| `height`   | Height of the bounding box (normalized)                |

## Example `.txt` Annotation File

For an image named `frame_001.jpg`, the annotation file `frame_001.txt` might contain:

```
0 0.512 0.634 0.087 0.129
```

This means:

-   The object is of class `0` (e.g., "ball")
-   It is centered at 51.2% width and 63.4% height of the image
-   The box covers 8.7% of the image width and 12.9% of the height

## One Image = One Annotation File

For each image in your dataset:

-   There is **one `.txt` file** with the same name (but `.txt` extension)
-   If no objects are present, the file may be empty or not exist at all
-   Multiple lines = multiple objects in the same image

---

### 6.3 Implementation

We have a folder containing our 1100 images and another folder containing our 1100 annotations.
We're going to split them using the [`splitTrainValFolder.py`](scripts/detection/splitTrainValFolder.py) script, as follows:

dataset/
├── train/
│ ├── images/
│ └── labels/
├── val/
│ ├── images/
│ └── labels/

Let's start training

```
yolo detect train \
  model=yolov8n.pt \
  data=... \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  name=yolo_n_1100 \
  device=mps
```

### 6.4 Resultats

[`Consult here`](models/detection/yolo_n_1100/)

#### Confusion Matrix (Normalized)

-   **True Positives (ball predicted as ball)**: 97%
-   **False Negatives (ball predicted as background)**: 3%
-   **True Negatives (background predicted as background)**: 100%
-   **False Positives (background predicted as ball)**: 0%

The model demonstrates very high accuracy, especially in distinguishing "ball" from "background", with minimal misclassification.

---

#### Confusion Matrix (Absolute)

|                           | True: ball | True: background |
| ------------------------- | ---------- | ---------------- |
| **Predicted: ball**       | 213        | 12               |
| **Predicted: background** | 7          | N/A              |

-   213 correct detections of "ball"
-   7 missed detections (false negatives)
-   12 false detections (false positives)

---

#### F1-Confidence Curve

-   **Optimal confidence threshold**: 0.41
-   **F1 Score plateau**: Around 0.96  
    Indicates a good balance between precision and recall.

---

#### Label Distribution

-   Object positions are broadly distributed in the frame.
-   Bounding boxes are generally small, which is expected for objects like balls.

---

#### Precision-Confidence Curve

-   Precision remains consistently high across confidence levels.
-   At a threshold of **0.73**, precision reaches **100%**, indicating no false positives.

---

#### Precision-Recall Curve

-   **Precision**: 0.986
-   **Recall**: 0.99  
    Excellent tradeoff between correctly identifying balls and avoiding false alarms.

---

#### Loss Curves

| Metric                | Description                                               |
| --------------------- | --------------------------------------------------------- |
| **Train Box Loss**    | Steady decrease, showing the model is learning locations. |
| **Train Cls Loss**    | Rapid drop, then plateau—typical for classification loss. |
| **Train DFL Loss**    | Improving localization quality over epochs.               |
| **Validation Losses** | Mirrors training trends—minimal overfitting.              |

---

#### Metrics Across Epochs

-   Precision and recall consistently improve.
-   **mAP@0.5** > 0.95
-   **mAP@0.5:0.95** ≈ 0.70

Indicates strong generalization and high detection performance.

---

#### Final Remarks

This model shows excellent performance:

-   High F1 score, precision, and recall
-   Low error rates
-   Strong generalization

It is suitable for deployment or as a base model for pseudo-labeling and dataset expansion.

### 6.5 Application on our dataset

We can then apply our model to the remaining photos in batch_00; it will pre-annotate the images.
We use the [`detectBall.py`](scripts/detection/detectBall.py) script.

## And what’s next?

You can then repeat this action on more and more images, in order to refine your model, remembering to adapt Yolov8's version.

### Model on 4256 images

A model made on batch_00.
[`Consult here`](models/detection/yolo_s_4256/)

```
yolo detect train \
  model=yolov8s.pt \
  data=... \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  name=yolo_s_4256 \
  device=mps
```

#### Confusion Matrix (Normalized)

|                      | True: Ball | True: Background |
| -------------------- | ---------- | ---------------- |
| **Pred: Ball**       | 0.95       | 1.00             |
| **Pred: Background** | 0.05       | 0.00             |

**Interpretation:**  
The model correctly classifies 95% of ball instances and 100% of background instances. Only 5% of ball instances are misclassified as background.

---

#### Confusion Matrix (Absolute)

|                      | True: Ball | True: Background |
| -------------------- | ---------- | ---------------- |
| **Pred: Ball**       | 595        | 117              |
| **Pred: Background** | 34         | 0                |

**Interpretation:**  
Out of 629 actual ball instances, 595 were correctly predicted, and 34 were missed. Among 117 background instances, all were falsely predicted as "ball", indicating some confusion or lack of negative examples.

---

#### F1-Confidence Curve

-   The F1-score peaks around **0.90** at a confidence threshold of approximately **0.616**.
-   This suggests that a confidence threshold around 0.6 provides the best balance between precision and recall.

---

#### Label Distribution and Correlation

-   The `labels_correlogram` and `labels` plots indicate that the **"ball"** class is the only one annotated.
-   The spatial distribution of boxes shows strong consistency in the ball's location, with a tight correlation between width and height.

---

#### Precision-Confidence Curve

-   Precision reaches close to **1.0** at high confidence levels.
-   Optimal threshold (**0.945**) achieves **100% precision** across predictions.

---

#### Precision-Recall Curve

-   **mAP@0.5: 0.949** for the "ball" class.
-   Precision and recall both remain high until a steep drop at the end, confirming high model reliability for this class.

---

#### Recall-Confidence Curve

-   Recall is very high (close to **0.98**) across most confidence levels.
-   Significant drop only occurs at very high confidence thresholds (> 0.9).

---

#### Training and Validation Curves

##### Training Losses

-   **Box Loss** decreases steadily and smoothly from ~1.2 to ~0.6.
-   **Classification Loss** shows a strong initial drop and continues to decline.
-   **DFL (Distribution Focal Loss)** also decreases smoothly.

##### Validation Losses

-   Follow a similar trend to training losses, indicating no major overfitting.

##### Metrics

-   **Precision (B):** climbs towards ~0.95.
-   **Recall (B):** improves from 0.65 to over 0.90.
-   **mAP@0.5:** approaches 0.95.
-   **mAP@0.5:0.95:** rises from ~0.5 to ~0.75.

---

**Conclusion:**  
Your model shows strong learning progression and generalization ability. Minor adjustments in thresholding could enhance performance further. You can confidently use this model for production or pseudo-labeling tasks.

### Model on 9310 images

A model made on batch_00 and batch_01.
[`Consult here`](models/detection/yolo_s_9310/)

```
yolo detect train \
  model=yolov8s.pt \
  data=... \
  epochs=84 \
  imgsz=640 \
  batch=16 \
  name=yolo_s_9310 \
  device=mps
```

#### Confusion Matrix (Normalized)

-   The model predicts the `ball` class correctly 94% of the time and misclassifies it as `background` 6% of the time.
-   The `background` class is predicted perfectly with 100% accuracy.
-   The model shows a strong ability to differentiate between the two classes, with a slight confusion in identifying `ball` objects.

#### Confusion Matrix (Absolute Counts)

|            | True: ball | True: background |
| ---------- | ---------- | ---------------- |
| Pred: ball | 1025       | 151              |
| Pred: bg   | 61         | -                |

-   There are 151 false positives where background is misclassified as a ball.
-   There are 61 false negatives where balls are missed and labeled as background.

#### F1-Confidence Curve

-   The optimal confidence threshold for maximum F1 score (0.91) is at **0.311**.
-   Beyond this threshold, F1 performance starts to decline, indicating overconfidence leads to missed detections.

#### Precision-Confidence Curve

-   Precision reaches its peak at a confidence threshold of **0.959**.
-   At high confidence thresholds, only the most certain predictions are retained, which improves precision at the expense of recall.

#### Recall-Confidence Curve

-   Recall is highest at very low thresholds and starts decreasing beyond a confidence of **~0.8**.
-   Indicates the model is very sensitive to low-confidence detections.

#### Precision-Recall Curve

-   Precision and recall balance well across the threshold range.
-   The model achieves an **mAP@0.5 of 0.956**, which is very strong performance.

#### Labels Distribution and Correlogram

-   The label correlogram shows a balanced spatial distribution of detections (x, y).
-   Most `ball` instances have small bounding boxes (width and height concentrated under 0.05), which reflects the object’s typical appearance in images.

#### Metrics from Training Logs

##### Loss Curves

-   All loss types (`box_loss`, `cls_loss`, `dfl_loss`) steadily decrease during training and validation.
-   This is a strong indicator of good convergence without signs of overfitting.

##### Metric Evolution

-   Precision and recall steadily increase across epochs, stabilizing at high levels.
-   Final validation metrics:
    -   **Precision(B)**: ~0.91
    -   **Recall(B)**: ~0.94
    -   **mAP50**: ~0.96
    -   **mAP50-95**: ~0.80

#### Summary

-   The model performs very well on both detection and classification fronts.
-   Excellent mAP and F1 score suggest reliable object localization and classification.
-   Improvements could be made by reducing false positives on background data.
