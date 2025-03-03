# SIGMA: Sheaf-Informed Geometric Multi-Agent Pathfinding
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0.0-%23EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of [SIGMA](https://arxiv.org/abs/2502.06440) algorithm accepted for oral presentation at ICRA 2025. The framework integrates sheaf theory with multi-agent reinforcement learning for efficient path planning.

![Demo](./images/demo.gif)

## Requirements
```bash
conda create -n sigma python==3.10
conda activate sigma
pip install numpy==1.23.5
# CUDA 11.8
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Project Structure
```
.
├── configs.py        # Configuration parameters
├── model.py         # Core network architecture
├── train.py         # Distributed training entry
├── test.py          # Evaluation and visualization
├── environment.py   # Multi-agent simulation environment
└── worker.py        # Ray parallelization components
```

## Environment Types
SIGMA supports two types of environments:
1. **Room-like Environment**: Open spaces with obstacles simulating indoor scenarios
2. **Maze Environment**: Complex corridor structures with multiple pathways

You can switch between these environments by modifying parameters in `configs.py`. We provide pre-trained models for both environment types.

## Training
```bash
# Start training
python train.py
```

## Citation
If you use this work in your research, please cite:
```bibtex
@article{liao2025sigma,
  title={SIGMA: Sheaf-Informed Geometric Multi-Agent Pathfinding},
  author={Liao, Shuhao and Xia, Weihang and Cao, Yuhong and Dai, Weiheng and He, Chengyang and Wu, Wenjun and Sartoretti, Guillaume},
  journal={arXiv preprint arXiv:2502.06440},
  year={2025}
}
```

## Authors
Shuhao Liao  
Weihang Xia  
Yuhong Cao  
Weiheng Dai  
Chengyang He  
Wenjun Wu  
Guillaume Sartoretti
