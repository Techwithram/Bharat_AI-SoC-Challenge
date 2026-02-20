# 🗺️ Device Tree & Firmware Generation

![OS](https://img.shields.io/badge/OS-Ubuntu_22.04_LTS-orange)
![Tool](https://img.shields.io/badge/Tool-XSCT_|_Bootgen-blue)
![Target](https://img.shields.io/badge/Target-Kria_KV260-brightgreen)

This directory outlines the firmware generation workflow required to deploy our custom Programmable Logic (PL) on the AMD/Xilinx Kria KV260. 

Because we are running a dynamic Ubuntu Linux environment rather than a static bare-metal system, the custom hardware (Temporal Gate and MobileNetV2 IP) must be loaded dynamically at runtime without rebooting the board. This requires packaging the Vivado output into a specific format that the Kria's `xmutil` platform management utility can read.

---

<br/>


## 1. Bitstream Conversion (`.bit.bin`)

Standard Vivado exports a `.bit` file. However, the Linux FPGA manager requires this bitstream to be in a raw binary format. 

**How it is generated:**
We use the Xilinx `bootgen` utility. `bootgen` reads a simple Boot Image Format (`.bif`) text file that points to the raw bitstream and packages it for the Zynq UltraScale+ architecture.

**The Command:**
```bash
# firmware.bif contains: all: { [destination_device = pl] design_1_wrapper.bit }
bootgen -image firmware.bif -arch zynqmp -o custom_fruit_accel.bit.bin -w
```

---

<br/>


## 2. Device Tree Generation (dtbo file)
This is executed using the Xilinx Software Command-Line Tool (XSCT) from Vivado 2022.2. To bypass known Windows path-parsing bugs during generation, we navigate directly to the workspace and utilize a locally cloned device-tree-xlnx repository

**The Command:**

```bash
# 1. Navigate directly to the project directory
cd C:/Users/krish/Hardware_new

# 2. Define the workspace scratchpad
setws C:/Users/krish/Hardware_new/workspace

# 3. Generate the Device Tree Source using the local repository
createdts -hw design_1_wrapper.xsa -local-repo C:/device-tree-xlnx -out dts_output -overlay -platform-name kria_firmware
```
---
<br/>

## 3. Deployment of the device tree onto the FPGA board

To bring the hardware online, the Kria's platform management utility (xmutil) requires a specific folder structure containing three files:

 1. fruit_accel.bit.bin (The physical hardware)
 2. fruit_accel.dtbo (The memory map)
 3. shell.json (A configuration file defining the app as an XRT_FLAT shell)


**Commands :**

```bash
# 1. Create the system directory
sudo mkdir -p /lib/firmware/xilinx/fruit_accel

# 2. Move files into position
sudo cp custom_fruit_accel.bit.bin /lib/firmware/xilinx/fruit_accel/fruit_accel.bit.bin
sudo cp pl.dtbo /lib/firmware/xilinx/fruit_accel/fruit_accel.dtbo
sudo cp shell.json /lib/firmware/xilinx/fruit_accel/

# 3. Hot-swap the hardware accelerator
sudo xmutil unloadapp
sudo xmutil loadapp fruit_accel
```

<br/>

Upon a successful load, the Linux FPGA Manager flashes the bitstream into the Programmable Logic and merges the .dtbo into the live kernel. The AXI DMA and Temporal Gate IPs are now physically active and ready for inference
