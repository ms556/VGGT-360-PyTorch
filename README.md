# VGGT-360-PyTorch

An unofficial PyTorch implementation of **"VGGT-360: Geometry-Consistent Zero-Shot Panoramic Depth Estimation"**.

This repository provides a training-free framework that leverages VGGT-like 3D foundation models for high-quality, globally consistent depth estimation from $360^{\circ}$ equirectangular panoramas (ERP).

## Features
- **Uncertainty-Guided Adaptive Projection:** Dynamically allocates denser views to geometry-poor regions using Sobel gradient metrics.
- **Structure-Saliency Enhanced Attention:** Modifies VGGT's self-attention layers with structural priors (Log-Confidence Bias) to suppress hallucination artifacts without retraining.
- **Correlation-Weighted 3D Model Correction:** Blends fragmented multi-view depths seamlessly into an ERP depth map using intra-frame attention metrics (Sharpness, Locality, Symmetry).

## Installation

```bash
git clone [https://github.com/ms556/VGGT-360-PyTorch.git](https://github.com/ms556/VGGT-360-PyTorch.git)
cd VGGT-360-PyTorch
pip install -r requirements.txt