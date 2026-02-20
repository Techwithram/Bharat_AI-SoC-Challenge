# 🛠️ FPGA Hardware Architecture (Programmable Logic)

![Platform](https://img.shields.io/badge/Platform-Kria_KV260-blue)
![Architecture](https://img.shields.io/badge/Architecture-Zynq_UltraScale%2B-brightgreen)
![Toolchain](https://img.shields.io/badge/Toolchain-Vivado_|_Vitis_HLS_2022.2-orange)

This repository details the Programmable Logic (PL) hardware architecture developed by Team ZARCOS Automation for the Edge-Accelerated Fruit Ripeness Detection System. The design implements a high-throughput, power-gated AI pipeline utilizing an AXI Direct Memory Access (DMA) controller and custom hardware IP blocks.

---
<br/>


## ⚡ The Core Custom IPs

The hardware acceleration pipeline is split into two custom IP blocks, both synthesized from C++ using Vitis HLS.

### A. The Temporal Gate IP (Smart Preprocessing)

Before any heavy neural network calculations occur, the incoming image stream must pass through the Temporal Gate. 
* **Function:** It performs localized frame-differencing. It compares the incoming pixel stream against a baseline state to detect if a fruit has physically entered the camera's Region of Interest (ROI).
* **The "Why":** In a factory, conveyor belts are often empty between batches. Continuously running an AI accelerator on an empty belt wastes massive amounts of power. The Temporal Gate acts as a hardware-level power switch; if no movement is detected, it drops the AXI Stream packets, keeping the downstream Neural Network IP idle.
* **Control:** The sensitivity threshold of the gate is controlled via an AXI-Lite interface mapped directly to the ARM processor.
<br/>

<br/>


### B. The MobileNetV2 Hardware Accelerator (Layers IP)

This block contains the highly parallelized, unrolled logic for the feature extraction phase of our classification model.
* **Function:** It executes Inputs → Block 8 of the MobileNetV2 architecture.
* **The "Why":** Standard ARM processors struggle with the massive matrix multiplications required by early convolutional layers. By mapping these specific layers to the FPGA's DSP slices, we drastically reduce inference latency. 
* **Data Handling:** It utilizes a purely streaming architecture. As pixels flow in from the Temporal Gate, convolutions are computed on the fly without needing to buffer the entire image in local block RAM.

---

<br/>


## 2. Architecture & Dataflow Wiring

<br/>



![iip](https://github.com/user-attachments/assets/9484005a-0456-4e17-8b22-6190ef2ce6d3)
*(Note: Vivado Block Design PDF/PNG here)*

<br/>


The system is wired using the AMBA AXI4 protocol, specifically leveraging **AXI4-Stream** for high-speed video data and **AXI4-Lite** for control registers.

1. **PS to PL (MM2S):** The Cortex-A53 processor writes the 128x128 cropped image frame to DDR memory. The AXI DMA (Memory Map to Stream) reads this data via a High-Performance (HP) port and converts it into a continuous AXI4-Stream.
2. **Through the Gate:** The stream enters the Temporal Gate. If the threshold is met, the gate passes the `TDATA` stream directly into the Layers IP.
3. **Hardware Inference:** The Layers IP processes the image, churning through the depthwise separable convolutions of Blocks 1-8. 
4. **PL to PS (S2MM):** The resulting heavily condensed feature map is pushed out as an AXI4-Stream. The AXI DMA (Stream to Memory Map) catches this stream. 
5. **Packet Boundary:** The Layers IP asserts the `TLAST` signal on the final byte of the feature map, signaling to the DMA that the packet is complete and triggering a hardware interrupt to the ARM processor to wake up the software tail.

<br/>

---

## 3. The Vivado 2022.2 Implementation Workflow

The physical hardware was generated using the following standard Xilinx design flow:

<br/>

### Phase 1: High-Level Synthesis (Vitis HLS)
1. Wrote the C++ algorithms for both the Temporal Gate and the MobileNetV2 layers.
2. Applied `#pragma HLS INTERFACE axis` to input and output ports to enforce AXI4-Stream protocols.
3. Applied `#pragma HLS INTERFACE s_axilite` to control variables (like the Temporal Gate threshold) to allow ARM CPU adjustments.
4. Synthesized the C++ into RTL (Verilog/VHDL) and exported both as packaged IP cores.

<br/>

### Phase 2: Block Design Integration (Vivado)

1. Created a new Vivado project targeting the Kria KV260 board.
2. Imported the custom Vitis HLS IP cores into the IP Catalog.
3. Added the **Zynq UltraScale+ MPSoC** block and the **AXI Direct Memory Access (DMA)** block.
4. Disabled "Scatter Gather Engine" on the DMA to ensure simple, direct memory transfers.
5. Wired the AXI-Stream ports (`M_AXIS_MM2S` → `Temporal Gate` → `Layers IP` → `S_AXIS_S2MM`).
6. Generated the HDL wrapper, ran Synthesis, Implementation, and generated the final Bitstream (`.bit`).
7. Exported the Hardware (`.xsa` file) including the bitstream.

---

<br/>

## 4. Device Tree & Firmware Generation (XSCT)
<br/>


Because the Kria board runs a dynamic Ubuntu environment rather than a static PetaLinux build, the hardware is loaded at runtime using `xmutil`. This requires generating a Device Tree Overlay (`.dtbo`) from the exported `.xsa`.

The following commands were executed in the **Vivado XSCT Console (2022.2)** to safely generate the hardware memory map, bypassing Windows path-parsing limitations:

```tcl
# 1. Navigate directly to the project directory
cd C:/Users/krish/Hardware_new

# 2. Define a workspace to prevent generation errors
setws C:/Users/krish/Hardware_new/workspace

# 3. Generate the Device Tree Source (.dtsi) using a local Xilinx repository
createdts -hw design_1_wrapper.xsa -local-repo C:/device-tree-xlnx -out dts_output -overlay -platform-name kria_firmware
