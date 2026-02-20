import os
import mmap
import time
import struct
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# --- 1. HARDWARE MEMORY MAP CONFIGURATION ---
AXI_DMA_BASE = 0x40400000  # Verify this in your pl.dtsi!
DMA_SIZE = 0x10000

# Standard offsets for Xilinx AXI DMA
MM2S_CR = 0x00   # Control Register
MM2S_SA = 0x18   # Source Address
MM2S_LEN = 0x28  # Transfer Length
S2MM_CR = 0x30   # Control Register
S2MM_DA = 0x48   # Destination Address
S2MM_LEN = 0x58  # Receive Length

# Shared memory addresses in DDR for the DMA to read/write 
SRC_MEM_ADDR = 0x10000000 
DST_MEM_ADDR = 0x10100000 
MEM_SIZE = 0x1000000 # 16MB allocation

CLASS_NAMES = [
    'apple-overripe', 'apple-ripe', 'apple-rotten', 'apple-unripe',
    'banana-overripe', 'banana-ripe', 'banana-rotten', 'banana-unripe',
    'mango-overripe', 'mango-ripe', 'mango-rotten', 'mango-unripe',
    'orange-ripe', 'orange-rotten'
]

def write_reg(mem, offset, val):
    mem.seek(offset)
    mem.write(struct.pack('<L', val))

def read_reg(mem, offset):
    mem.seek(offset)
    return struct.unpack('<L', mem.read(4))[0]

def main():
    print("Loading ARM Software Tail...")
    interpreter = tflite.Interpreter(model_path="tail_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Calculate exact byte size expected by your tail model
    # (e.g., if tail input is 16x16x320 float32, bytes = 16*16*320*4)
    hw_output_shape = input_details[0]['shape']
    hw_output_bytes = np.prod(hw_output_shape) * 4 

    print("Mapping Physical Memory via /dev/mem...")
    f = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    
    # Map DMA Control Registers
    dma_mem = mmap.mmap(f, DMA_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=AXI_DMA_BASE)
    
    # Map Source (Image) and Dest (Feature Map) Memory
    src_mem = mmap.mmap(f, MEM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=SRC_MEM_ADDR)
    dst_mem = mmap.mmap(f, MEM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=DST_MEM_ADDR)
    
    # Turn on DMA Channels (Set Run/Stop bit)
    write_reg(dma_mem, MM2S_CR, 1)
    write_reg(dma_mem, S2MM_CR, 1)
    
    cap = cv2.VideoCapture(0)
    print("Starting Edge Inference. Press 'q' to quit.")
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        # --- ROI EXTRACTION (Computer Vision) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 2000:
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Crop and format for hardware
                roi = frame[y:y+h, x:x+w]
                resized_roi = cv2.resize(roi, (128, 128))
                img_array = (resized_roi / 255.0).astype(np.float32)
                
                # --- HARDWARE ACCELERATION (FPGA) ---
                # 1. Write image directly into physical DDR memory
                src_mem.seek(0)
                src_mem.write(img_array.tobytes())
                
                # 2. Setup DMA Receive Channel
                write_reg(dma_mem, S2MM_DA, DST_MEM_ADDR) 
                write_reg(dma_mem, S2MM_LEN, hw_output_bytes) 
                
                # 3. Setup DMA Transmit Channel & FIRE!
                write_reg(dma_mem, MM2S_SA, SRC_MEM_ADDR) 
                write_reg(dma_mem, MM2S_LEN, img_array.nbytes) 
                
                # 4. Wait for DMA S2MM channel to finish (Check idle bit)
                while not (read_reg(dma_mem, S2MM_CR) & 0x02):
                    pass
                
                # 5. Read hardware output map back from physical memory
                dst_mem.seek(0)
                hw_feature_map = np.frombuffer(dst_mem.read(hw_output_bytes), dtype=np.float32).reshape(hw_output_shape)
                
                # --- SOFTWARE TAIL (ARM) ---
                interpreter.set_tensor(input_details[0]['index'], hw_feature_map)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])
                
                # Display Results
                predicted_index = np.argmax(predictions[0])
                label = f"{CLASS_NAMES[predicted_index]}: {predictions[0][predicted_index]*100:.1f}%"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        # Calculate Hardware Co-Design FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"HW Accel FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("ZARCOS Live Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up memory mapping safely
    cap.release()
    cv2.destroyAllWindows()
    dma_mem.close()
    src_mem.close()
    dst_mem.close()
    os.close(f)

if __name__ == '__main__':
    main()
