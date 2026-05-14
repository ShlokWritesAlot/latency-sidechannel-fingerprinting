# Latency Side-Channel Fingerprinting of Neural Network Architectures

This repository contains the code, data, and research paper for the study: **"Latency Side-Channel Fingerprinting of Neural Network Architectures via Inference Timing Analysis"**.

## Overview
Latency-based side-channel leakage has emerged as a potential security risk in modern machine learning deployment environments. This research investigates whether inference latency can be exploited to fingerprint neural network architectures in black-box settings. By analyzing architectures such as VGG, ResNet, and Inception across CPU and GPU environments, we demonstrate that distinct computational structures produce distinguishable timing signatures. Our findings highlight the security implications of timing leakage, proving high-accuracy architecture identification using only statistical analysis of latency.

## Project Architecture

```
Research Paper 1/
├── Experiments/
│   ├── EXPERIMENT 1 - LATENCY MEASUREMENT + REPORT/      # Generates baseline execution times
│   ├── EXPERIMENT 2 - LATENCY DISTRIBUTION + VISUAL REPORT/# Statistical distribution & visualization
│   ├── EXPERIMENT 3 - MODEL CLASSIFICATION FROM LATENCY/ # PCA & Machine learning classifiers
│   ├── EXPERIMENT 4 - QUERY COUNT vs ATTACK SUCCESS/     # Sliding window analysis 
│   ├── EXPERIMENT 5 - NOISE ROBUSTNESS/                  # Classifier stability under network jitter
│   └── EXPERIMENT 6 - HETEROGENEOUS HARDWARE/            # CPU vs GPU latency signature comparison
├── Paper/
│   ├── paper.tex                                         # LaTeX source for the manuscript
│   └── Latency_Side_Channel_Fingerprinting.pdf           # Compiled manuscript
├── Results/                                              # Comprehensive CSV metrics and high-res plots
├── mock_gpu.py                                           # Emulates GPU runtime characteristics
└── requirements.txt                                      # Project dependencies
```

## Methodology

We define a 6-phase experimental pipeline to rigorously evaluate the feasibility and robustness of latency fingerprinting:

1. **Measurement**: Collecting high-precision inference latencies for `resnet18`, `resnet50`, `vgg16`, and `inception_v3`.
2. **Distribution**: Analyzing the statistical variance (mean, std, skewness, kurtosis) to establish unique architectural signatures.
3. **Classification**: Training supervised classifiers (KNN, Random Forest, Logistic Regression) using stratified 5-fold cross-validation, visualizing separation via PCA.
4. **Query Windowing**: Using a sliding window approach to evaluate how consecutive queries improve attack confidence.
5. **Robustness**: Injecting artificial system noise (jitter) to stress-test the classification pipeline under realistic constraints.
6. **Heterogeneous Hardware**: Comparing the latency side-channel on CPUs against highly-parallel GPU architectures, capturing synchronization overheads.

## Key Findings
- **High-Accuracy Fingerprinting**: Latency alone is sufficient to accurately classify network architectures (>90% accuracy in controlled environments).
- **Architectural Signatures**: Deep but sequential models (e.g., VGG-16) display high latency and variance, whereas highly optimized or shallow models (ResNet-18) group tightly in the latent space.
- **Hardware Variation**: GPU environments exhibit distinct synchronization artifacts, altering the baseline latency profile compared to CPU execution, yet architectural fingerprinting remains feasible.
- **Robustness Limitations**: While accuracy scales positively with observation window size, substantial system noise can effectively mitigate the timing side-channel.

## Setup and Execution
To replicate the study, install the required dependencies:

```bash
pip install -r requirements.txt
```

Execute individual experiments from the root directory to automatically populate the `Results/` directory with detailed CSVs and plots.

## Author
**Shlok Pandey**  
Manipal University Jaipur  
Email: shlokpandey8219@gmail.com
