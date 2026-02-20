# 🧠 Model Training & Architecture (CNN_ARM)

<br/>


![Model](https://img.shields.io/badge/Model-MobileNetV2-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%2Fkeras-FF6F00)
![Environment](https://img.shields.io/badge/Trained_on-Kaggle-20BEFF)
![Task](https://img.shields.io/badge/Task-Image_Classification-success)

This folder contains the software-side assets, training scripts, and the exported model files for the fruit ripeness classification system. The model is specifically optimized for hardware/software co-design on the Kria KV260 edge accelerator.

<br/>


---

## 📊 1. The Training Dataset

The model is trained on a comprehensive custom dataset of agricultural produce, specifically curated to detect both the type of fruit and its exact ripeness stage. 

**💪 Supported Fruits:**
* 🍎 Apple
* 🍌 Banana
* 🥭 Mango
* 🍊 Orange

**🪜 Classification Stages (Per Fruit):**
* **Unripe:** Green, hard, or visually underdeveloped.
* **Ripe:** Optimal color and texture for consumption/processing.
* **Overripe:** Early signs of degradation, bruising, or excessive softness.
* **Rotten:** Severe degradation, mold, or spoilage.

The dataset is organized hierarchically into directories (e.g., `apple-ripe`, `mango-rotten`), allowing the `ImageDataGenerator` or `image_dataset_from_directory` utilities to automatically infer the 14 distinct class labels during training.

---
<br/>

## 🛠️ 2. The Model: MobileNetV2



To achieve the ultra-low latency required for an industrial conveyor belt while staying within the resource constraints of an edge device, we selected **MobileNetV2** as the backbone architecture. 

Developed by researchers at Google, MobileNetV2 is an incredibly efficient convolutional neural network designed specifically for mobile and embedded vision applications. 

**Key Architectural Advantages:**
* **Depthwise Separable Convolutions:** Instead of standard heavy convolutions, it splits the operation into a depthwise convolution and a 1x1 pointwise convolution, drastically reducing the required number of parameters and computations.
* **Inverted Residuals:** It expands the channel count in the middle of the blocks to capture complex features, and then shrinks them back down, which preserves memory while maintaining high accuracy.
* **Linear Bottlenecks:** It removes non-linear activation functions in the narrow layers to prevent the loss of critical information.
* **Ultra-Lightweight:** The base network utilizes roughly 3.4 million parameters and only requires about 300 million multiply-add operations, making it exponentially faster than architectures like ResNet or VGG.
* **Hardware Friendly:** It utilizes the ReLU6 activation function, which bounds the activation values between 0 and 6, making it highly optimized for fixed-point quantization and edge-device computation.

<br/>

---

## ☁️ 3. Kaggle Deployment

<br/>


The entire training Dataset has been moved to the Kaggle for delopying it globally , the dataset consists of 30.4k Images consisiting of 14 different classes of 4 different fuits namely Apple, Mango, Banana, Orange.  
*Note: For edge deployment, this `.h5` model is later sliced at `block_9_expand` into a `.tflite` format, allowing the Kria KV260 ARM processor to smoothly catch the hardware DMA output.*

<br/>

---
[Click here for viewing the dataset and the AI model](https://www.kaggle.com/datasets/ramkrishthatikonda/bharat-ai-soc-challenge-ps5)
---

<br/>

## Model Prediction and Model Metrics
<br/>

The prediction Metrics across all the classes

### Classification Report


| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **apple-overripe** | 0.90 | 1.00 | 0.95 | 92 |
| **apple-ripe** | 1.00 | 0.93 | 0.96 | 683 |
| **apple-rotten** | 0.98 | 1.00 | 0.99 | 400 |
| **apple-unripe** | 0.92 | 1.00 | 0.96 | 314 |
| **banana-overripe** | 0.93 | 0.99 | 0.96 | 113 |
| **banana-ripe** | 0.98 | 0.97 | 0.97 | 154 |
| **banana-rotten** | 1.00 | 0.93 | 0.96 | 185 |
| **banana-unripe** | 0.94 | 0.99 | 0.96 | 110 |
| **mango-overripe** | 1.00 | 0.89 | 0.94 | 725 |
| **mango-ripe** | 0.67 | 0.66 | 0.66 | 321 |
| **mango-rotten** | 1.00 | 0.55 | 0.71 | 274 |
| **mango-unripe** | 0.59 | 0.99 | 0.74 | 329 |
| **orange-ripe** | 0.99 | 0.98 | 0.98 | 388 |
| **orange-rotten** | 1.00 | 0.97 | 0.98 | 176 |
| | | | | |
| **Accuracy** | | | **0.91** | **4264** |
| **Macro Avg** | 0.92 | 0.92 | 0.91 | 4264 |
| **Weighted Avg** | 0.93 | 0.91 | 0.91 | 4264 |


## Confusion Matrix:
 The confusion matrix defines the prediction of the images based on the train and the test images split that is used during the training of the model.
 
 <br/>
 
 
 <img width="1919" height="1023" alt="Screenshot 2026-02-20 161651" src="https://github.com/user-attachments/assets/6276f275-0ff4-4619-ae38-f0fa6f62bf13" />

<br/>

