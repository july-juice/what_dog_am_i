# Facial Expression Recognition Project

This project explores facial detection and expression recognition using OpenCV and custom deep learning models based on ResNet50.  
It combines computer vision (for face tracking) with transfer learning (for emotion classification).

---

## Overview

- Face detection handled by OpenCV Haar Cascade face detector.
- Each webcam frame is converted to grayscale for faster processing.
- The detector identifies all faces in the frame and selects the largest one for further analysis.
- A custom-trained deep learning model (ResNet50 backbone) is used to classify facial expressions.

---

## Detection Logic

1. Convert webcam frame to grayscale.  
2. Detect faces using Haar Cascade.  
3. Pick the largest detected face.  
4. Pass that face region to a deep learning model for expression classification.

---

## Model Development

### Attempt 1 — FER (Failed)
- Tried the [FER](https://pypi.org/project/fer/) model for emotion recognition.
- Rejected due to compatibility issues between tensorflow, opencv-python and numpy.

---

### Attempt 2 — Custom Model (`CustomNetworkv1`)
- Created a custom dataset using random internet images *(for personal, non-commercial use)*.
- Trained a model using ResNet50 with transfer learning.
- **Performance:**
  - Train accuracy: **94%**
  - Validation accuracy: **84%**
  - Test accuracy: **79%**

---

### Attempt 3 — Refined Dataset (`CustomNetworkv2`)
- Improved dataset by reducing class overlap (e.g., removing “smile” and “perplexed”).
- Cropped all images to focus only on face regions.
- Test accuracy: **79.66%**, loss: `0.6466`.

---

### Attempt 4 — Fine-Tuning (overwritten `CustomNetworkv2`)
- Unfroze deeper layers of ResNet50 for fine-tuning.
- Improved test accuracy to **88.24%**.
- “Scream” face became especially accurate (almost *too* sensitive).

---

### Attempt 5 — Data Augmentation (`CustomNetworkv3`)
- Wrote a small webcam capture script to automatically take photos every two seconds.
- Used this to collect new training images for dataset expansion.
- Final test accuracy: **~89%**, loss: `0.245`.

---

## Issues

- **Face detection intermittency:**  
  Haar Cascade sometimes fails to detect faces reliably. Improved by:
  - Decreasing `scaleFactor` (higher sensitivity, slower speed)
  - Decreasing `minSize` from `(30, 30)` to `(10, 10)`

- **Class inconsistency:** 
  “Tongue out” expression detects intermittently (works barely for me, almost never for others).

- **Window freeze:**  
  `cv2.destroyAllWindows()` occasionally freezes the Python kernel on macOS.  
  Temporary fix: manually restart the kernel after quitting the app.

- **Webcam behavior on macOS:**  
  The webcam sometimes connects to iPhone Continuity Camera instead of the Mac’s built-in camera.

---

## Conclusions

- Transfer learning can achieve high accuracy with limited data.  
- Dataset curation (cleaning overlapping categories) dramatically improves model reliability.  
- Fine-tuning deeper layers of pretrained networks provides significant performance gains.  
- Practical integration (webcam + live classification) highlights real-world challenges in face detection reliability.

---

## Future Work

- Replace Haar Cascade with a more robust face detector (e.g., YOLOv8 face).  
- Add automatic data labeling for new captures.  
- Implement real-time smoothing/filtering of predictions to avoid flickering outputs.  
- Deploy as a small desktop or web app.

---

## Summary

- This was a fun, hands-on project to explore computer vision, transfer learning, and automation — combining OpenCV, TensorFlow, and custom dataset creation to teach a model to “read” facial expressions.

Dataset currently unavailable - will be made available-upon-request at a later date.
