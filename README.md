# EHA-YOLOv8: Enhanced Headlight-Aware YOLOv8
### Lightweight Nighttime Detection for Edge Devices

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)
![YOLOv8](https://img.shields.io/badge/Base-YOLOv8-pink)
![Deployment](https://img.shields.io/badge/Device-Jetson%2FEdge-green)

## 📖 Overview

**EHA-YOLOv8** is a specialized extension of the YOLOv8 architecture designed to tackle the unique challenges of **nighttime road detection**.



Standard detectors often struggle in two specific nighttime scenarios:
1.  **Headlight Glare:** Blown-out regions caused by oncoming traffic.
2.  **Low-Light Zones:** Loss of structural information in non-illuminated areas.

EHA-YOLOv8 introduces a **Headlight-Aware Attention Module** that generates an auxiliary mask to identify and suppress glare regions while enhancing features in dark areas. Unlike complex spatio-temporal models, this architecture remains lightweight and is optimized for single-frame detection on embedded systems.

---

## 🚀 Key Features

* **Headlight-Aware Attention Module:** A lightweight convolutional branch that processes intermediate feature maps to create an auxiliary mask, flagging high-intensity glare regions.
* **Glare Suppression:** The network learns to suppress features in "blown-out" areas and amplify structural cues in dark areas using the generated mask.
* **Dual-Loss Strategy:** Combines standard YOLO detection loss with a **Mask-Based Auxiliary Loss** to guide the attention mechanism during training.
* **Edge-Native Design:** Optimized for low-power hardware (Nvidia Jetson, RPi) with support for channel pruning and INT8 quantization.
* **High-Efficiency Input:** Default training resolution of **416×416** ensures high FPS without sacrificing critical detection accuracy.

---

## 🏗️ Architecture

The architecture retains the speed of the YOLOv8 Backbone-Neck-Head structure but injects an attention mechanism to handle illumination variance.

<img width="1989" height="1490" alt="APP1" src="https://github.com/user-attachments/assets/5485aee7-23c1-4ba7-92e6-4b4f2d394894" />

<img width="453" height="497" alt="image" src="https://github.com/user-attachments/assets/ee19ee0f-3fe9-484c-81f8-9a7b44518609" />

```mermaid
graph TD
    Input[Input Image 416x416] --> Backbone[CSPDarknet Backbone]
    
    subgraph "Feature Extraction"
    Backbone --> Features[Intermediate Features]
    end

    subgraph "Headlight Attention Branch"
    Features --> ConvBranch[Small Conv Branch]
    ConvBranch --> AuxMask[Auxiliary Glare Mask]
    AuxMask -->|Auxiliary Loss| Training[Loss Calculation]
    end

    subgraph "Detection Path"
    Features --> Fusion[Attention Fusion]
    Fusion --> Neck[FPN + PAN Neck]
    Neck --> Head[YOLOv8 Head]
    Head --> Output[BBox + Class + Objectness]
    end
