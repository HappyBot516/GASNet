# GASNet: Progressive Resolution-Aware Supervision and Gabor Guidance for Accurate Liver Vessel Segmentation

GASNet is a novel deep learning framework designed for high-precision liver vessel segmentation in CT scans. It addresses challenges such as fine vessel continuity, scale variance, and low contrast by integrating three key components:

- An **adaptive ConvNeXt backbone** for hierarchical feature extraction,
- A **learnable Gabor multi-filter module** for vessel-aware structural guidance,
- A **progressive resolution-aware supervision** mechanism for deep multi-scale learning.

> 📄 This repository supports code implementation for the GASNet architecture as proposed in our paper:
>
> **GASNet: Progressive Resolution-Aware Supervision and Gabor Guidance for Accurate Liver Vessel Segmentation**  
> Qing Yang, Gang Wang, Xiangyu Meng, Huanhuan Dai, Hanyu Wang, Wenqian Yu, Xun Wang*  

---

## 🔧 Installation

GASNet is implemented in **PyTorch** and tested with:

- Python 3.8
- PyTorch ≥ 1.10
- torchvision ≥ 0.11
- CUDA 11.1+

```bash
git clone https://github.com/yourname/GASNet.git
cd GASNet
pip install -r requirements.txt
