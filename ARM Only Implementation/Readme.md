# 💻 Software-Only Inference (ARM Cortex-A53 Baseline)

![Platform](https://img.shields.io/badge/Platform-Kria_KV260-blue)
![Processor](https://img.shields.io/badge/Processor-ARM_Cortex--A53-red)
![Framework](https://img.shields.io/badge/Framework-TFLite_Runtime-FF6F00)
![Task](https://img.shields.io/badge/Task-Image_Classification-success)

This directory contains the software-only implementation of our Fruit Ripeness Detection model. In this setup, the entire MobileNetV2 pipeline (Layers 1 through the Dense output) is executed purely in software on the Kria KV260's quad-core ARM Cortex-A53 processor.

This implementation serves two critical purposes:
1. **Verification:** It proves the end-to-end mathematical accuracy of the trained deep learning model in a real-world edge environment.
2. **Benchmarking:** It establishes a baseline CPU inference latency (Frames Per Second) to directly compare against our custom FPGA hardware accelerator.

---

## ⚙️ The Software Stack

Running full TensorFlow/Keras (`.h5` files) on an embedded ARM processor is highly inefficient and causes thermal throttling. To optimize this CPU baseline, the entire Keras model was converted into a **TensorFlow Lite (`.tflite`)** flatbuffer. 



We utilize the lightweight `tflite_runtime` library instead of the massive standard TensorFlow package, which saves memory and allows the Kria board to dedicate maximum CPU cycles to the actual matrix multiplications.

---

## 🔄 The Inference Pipeline (Step-by-Step)

The `arm_inference.py` script acts as the complete edge AI pipeline. Here is exactly how the data flows from the physical camera to the final prediction:

### 1. Image Capture
The script opens a video stream using standard OpenCV (`cv2.VideoCapture`). It continuously grabs raw BGR frames from the USB/MIPI camera connected to the Kria board.

### 2. Preprocessing & Formatting

Neural networks are extremely strict about the data they ingest. The raw 1080p camera frame cannot be fed directly into the model. The ARM processor applies the following transformations:
* **Resizing:** The model was trained on 128x128 pixel images. OpenCV uses `cv2.resize(frame, (128, 128))` to compress the visual data down to the exact expected tensor shape.
* **Normalization:** Raw pixel values range from 0 to 255. The script divides the array by 255.0 to scale the data to a `[0.0, 1.0]` float32 range, matching the exact normalization used during the Kaggle training phase.
* **Dimension Expansion:** The TFLite interpreter expects a batch dimension. `np.expand_dims` converts the `(128, 128, 3)` image into a `(1, 128, 128, 3)` tensor.

### 3. Model Execution
The preprocessed tensor is loaded into the TFLite Interpreter. The `interpreter.invoke()` command forces the ARM CPU to calculate all the MobileNetV2 depthwise separable convolutions sequentially.

### 4. Post-Processing & Display
The model outputs an array of 14 probabilities (one for each fruit/ripeness class).
* The script uses `np.argmax()` to find the index of the highest probability.
* This index is mapped to the `CLASS_NAMES` array to retrieve the human-readable string (e.g., `mango-ripe`).
* OpenCV draws a bounding box around the frame and overlays the textual prediction and confidence score before rendering the frame to the monitor.

---

## 🚀 Execution Instructions

**1. Prerequisites on the Kria KV260:**
```bash
pip3 install opencv-python numpy tflite-runtime
