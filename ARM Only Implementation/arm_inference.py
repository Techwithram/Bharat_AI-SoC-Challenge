import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite

# --- Configuration ---
MODEL_PATH = "fruit_ripeness_full.tflite"
IMG_SIZE = 128

CLASS_NAMES = [
    'apple-overripe', 'apple-ripe', 'apple-rotten', 'apple-unripe',
    'banana-overripe', 'banana-ripe', 'banana-rotten', 'banana-unripe',
    'mango-overripe', 'mango-ripe', 'mango-rotten', 'mango-unripe',
    'orange-ripe', 'orange-rotten'
]

def main():
    print("Loading Full ARM Software Model...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    cap = cv2.VideoCapture(0)
    print("Starting CPU Inference. Press 'q' to quit.")
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        height, width = frame.shape[:2]
        
        # --- PREPROCESSING ---
        # 1. Resize to match training data
        resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        
        # 2. Normalize to [0, 1] and typecast to float32
        img_array = (resized_frame / 255.0).astype(np.float32)
        
        # 3. Add batch dimension: (128, 128, 3) -> (1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # --- INFERENCE ---
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        # --- POST-PROCESSING ---
        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index]
        label = f"{CLASS_NAMES[predicted_index]}: {confidence*100:.1f}%"
        
        # --- DISPLAY ---
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 255, 0), 6)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (0, 0), (tw + 20, th + 30), (0, 255, 0), -1)
        cv2.putText(frame, label, (10, th + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        # Calculate and display CPU FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"ARM CPU FPS: {fps:.1f}", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("ZARCOS ARM Baseline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
