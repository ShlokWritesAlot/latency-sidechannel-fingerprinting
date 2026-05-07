# Latency Side-Channel Fingerprinting of Neural Network Architectures

This repository contains the code, data, and research paper for the study: **"Latency Side-Channel Fingerprinting of Neural Network Architectures via Inference Timing Analysis"**.

## Abstract
Latency-based side-channel leakage has emerged as a potential security risk in modern machine learning deployment environments. This research investigates whether inference latency can be exploited to fingerprint neural network architectures in black-box settings. By analyzing architectures like VGG, ResNet, and Inception, we demonstrate that distinct computational structures produce distinguishable timing signatures. Our findings highlight the security implications of timing leakage and demonstrate high-accuracy architecture identification using only latency information.

## Project Structure
```
Research Paper 1/
├── Experiments/
│   ├── EXPERIMENT 1 - LATENCY MEASUREMENT + REPORT/
│   │   └── latency_measurement.py      # Measures raw latency data
│   ├── EXPERIMENT 2 - LATENCY DISTRIBUTION + VISUAL REPORT/
│   │   └── latency_distribution.py     # Visualizes latency as histograms
│   ├── EXPERIMENT 3 - MODEL CLASSIFICATION FROM LATENCY/
│   │   └── model_classifier.py         # Trains KNN/LR to identify models
│   └── EXPERIMENT 4 - QUERY COUNT vs ATTACK SUCCESS/
│       └── query_count_vs_accuracy.py  # Evaluates accuracy vs number of queries
├── Paper/
│   ├── paper.tex                       # LaTeX source for the paper
│   └── Latency_Side_Channel_Fingerprinting...pdf # Compiled PDF
├── Results/
│   ├── accuracy_vs_query_count.png     # Experiment 4 visualization
│   ├── classification_report.csv       # Experiment 3 detailed metrics
│   ├── latency_distributions.png       # Experiment 2 visualization
│   ├── latency_results.csv             # Raw data from Experiment 1
│   └── latency_summary.csv             # Statistical summary of latencies
├── requirements.txt                    # Project dependencies
└── README.md                           # This file
```

## Setup and Installation
To run the experiments, ensure you have Python installed and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Experiments Overview

### 1. Latency Measurement
Measures the inference latency of `resnet18`, `resnet50`, `vgg16`, and `inception_v3` on a CPU. It runs 100 passes per model to compute mean, standard deviation, and P95 latency.

### 2. Latency Distribution
Generates histograms for each model's latency distribution to visualize the separation and overlap between different architectures.

### 3. Model Classification
Uses statistical features (mean, std, P95) to train K-Nearest Neighbors (KNN) and Logistic Regression (LR) classifiers. It evaluates the ability of an attacker to identify the model based on timing data.

### 4. Query Count vs. Attack Success
Analyzes how the accuracy of the fingerprinting attack improves as the attacker gathers more timing samples (N) per prediction.

## Results Summary
- **ResNet-18**: Highly distinguishable with low latency and low variance (~42ms).
- **VGG-16**: Highest latency and highest variability (~297ms).
- **Classification Performance**: KNN achieved ~86.25% accuracy, while Logistic Regression achieved ~80.00%.
- **Impact of Queries**: Accuracy increases as the number of queries per prediction increases, reaching ~87.5% with 10 queries.

## Author
**Shlok Pandey**  
Manipal University Jaipur  
Email: shlokpandey8219@gmail.com
