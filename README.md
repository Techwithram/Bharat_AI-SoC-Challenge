# 🍊 Edge-Accelerated Fruit Ripeness Detection System

![Platform](https://img.shields.io/badge/Platform-Kria_KV260-blue)
![OS](https://img.shields.io/badge/OS-Ubuntu_Linux-orange)
![Framework](https://img.shields.io/badge/Framework-TensorFlow_Lite-FF6F00)
![Hardware](https://img.shields.io/badge/Hardware-Custom_Temporal_IP-brightgreen)
![Bharat SoC Challenge](https://img,shields,io/badge/Bharat_SoC_Challenge)

A high-performance, bare-metal hardware/software co-design edge AI application for detecting fruit quality and ripeness (Apples, Oranges, Mangos, Bananas). Built natively on Ubuntu Linux for the AMD/Xilinx Kria KV260, this project partitions a custom MobileNetV2 architecture across Programmable Logic (PL) and the ARM Cortex-A53 processor to achieve ultra-low latency industrial sorting.

---

## 1. Problem Statement

The agricultural and food processing industry handles millions of tons of fruit daily. Currently, quality control and ripeness sorting rely heavily on manual human inspection or rudimentary optical sorting machines. Manual inspection is slow, highly subjective, and prone to fatigue-induced errors, leading to massive food waste and inconsistent product quality. As factory conveyor belts increase in speed to meet global supply chain demands, traditional sorting methods are becoming the primary bottleneck in agricultural production lines.



## 2. Industry Gaps & The Necessity of Edge AI

There is a massive gap in how modern fruit processing companies handle automated quality control. 

**The Flaws in Current Approaches:**
* **Basic Optical Sensors:** Standard color-sorting cameras cannot detect complex bruising, localized rot, or subtle ripeness indicators (like the transition of gradients on a mango).
* **Cloud-Connected AI:** Sending high-definition factory video feeds to cloud servers for AI analysis introduces catastrophic latency. On a fast-moving conveyor belt, a round-trip delay of even 200 milliseconds means the targeted fruit has already passed the mechanical sorting arm. Furthermore, streaming constant video requires massive bandwidth and poses a critical single point of failure—if the factory's internet drops, the entire production line stops.

**The Edge AI Mandate:**
To solve this, advanced inference must be brought directly to the edge—literally positioned over the conveyor belt. Edge AI is the *only* alternative that guarantees deterministic, real-time latency without relying on external network bandwidth. However, deploying heavy Convolutional Neural Networks (CNNs) on standard embedded processors (like Raspberry Pis or basic industrial PCs) results in thermal throttling and low frame rates. True real-time industrial sorting requires dedicated hardware acceleration.

## 3. The Proposed Solution: Heterogeneous Co-Design

This project completely bypasses the limitations of standard embedded processors by utilizing a custom hardware-software partitioned architecture on the Kria KV260 Vision AI Starter Kit.



Instead of running the entire AI model sequentially on a CPU, the workload is physically split:
* **The Hardware Accelerator (FPGA / PL):** A custom-designed datapath handles the heaviest mathematical operations. Image frames are streamed from Ubuntu Linux memory directly into the FPGA fabric via an AXI Direct Memory Access (DMA) controller. The PL executes the feature-extraction layers of our custom MobileNetV2 model (Inputs through Block 8) in parallel, dramatically reducing processing time.
* **The Software Inference (ARM CPU / PS):** The remaining classification layers (Block 9 through the Dense output) are packaged as a lightweight TensorFlow Lite model (`tail_model.tflite`). The ARM processor catches the heavily condensed feature map from the DMA and completes the final classification of the fruit's ripeness.

This co-design achieves the throughput of custom silicon while maintaining the flexibility of a software-based classification tail.

## 4. Uniqueness of the Solution: Smart Preprocessing

What separates this architecture from standard edge AI deployments is the highly optimized preprocessing pipeline designed specifically to minimize unnecessary compute cycles and DMA bandwidth usage.

### Region of Interest (ROI) Extraction
Rather than flooding the neural network with entire 1080p camera frames containing mostly empty conveyor belt space, the system utilizes a fast ROI extraction step. 
* It dynamically identifies the bounding box of the physical fruit.
* It crops and scales only the relevant pixel data to the 128x128 input required by the accelerator.
* This ensures the neural network focuses 100% of its parameters on the fruit's texture and color, drastically improving accuracy and reducing the memory payload sent over the AXI bus.



### Custom Temporal Gate IP
The absolute core innovation of this hardware design is the custom **Temporal Gate IP** placed before the Neural Network accelerator in the FPGA fabric. 
* In a factory setting, there are frequent gaps between fruits on a belt. Standard AI pipelines waste massive amounts of power and compute resources continuously evaluating empty frames.
* The Temporal Gate acts as a hardware-level trigger. It compares incoming pixel streams and only allows data to pass into the heavy MobileNetV2 IP block if a significant change (a new fruit entering the frame) is detected.
* If the frame is static, the gate remains closed, idling the downstream AI hardware and saving significant power, making this an incredibly green, energy-efficient solution for massive industrial scaling.

---
*Developed and maintained by Team ZARCOS Automation.*
