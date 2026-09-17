# SARA-Net — Multimodal SAR and Climate Data Fusion for Flood Risk Assessment in the Sundarbans

<p align="center">
  <img src="figures/discovery.png" alt="SARA-Net discovery analysis" width="100%">
</p>

<p align="center">
  <b>Sundarbans Adaptive Risk Assessment Network</b><br>
  A multimodal deep learning framework combining Sentinel-1 SAR imagery with climate context for flood-risk assessment.
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <img src="https://img.shields.io/badge/Remote%20Sensing-Sentinel--1-1f6feb" alt="Remote Sensing">
  <img src="https://img.shields.io/badge/Model-Multimodal%20Fusion-2ea44f" alt="Multimodal Fusion">
  <img src="https://img.shields.io/badge/XAI-LayerCAM-8b5cf6" alt="LayerCAM">
  <img src="https://img.shields.io/badge/Study%20Area-Sundarbans-0f766e" alt="Sundarbans">
</p>

---

## Introduction

The **Sundarbans**, shared by Bangladesh and India, is the world's largest mangrove forest and an important natural shield for coastal communities against cyclones, storm surges, and flooding. The region is exposed to increasing environmental pressure associated with climate change, sea-level rise, irregular precipitation, salinity changes, and hydrological stress.

Flood monitoring in this environment is challenging. Optical satellite observations can become unavailable during periods of persistent cloud cover, while rainfall-only or single-factor approaches cannot capture the full spatial and environmental context of flooding.

**SARA-Net (Sundarbans Adaptive Risk Assessment Network)** was developed as a multimodal deep learning framework that combines:

- **Sentinel-1 Synthetic Aperture Radar (SAR)** imagery for spatial and structural information
- **Rainfall and river-water-level data** for environmental context
- **Multimodal feature fusion** for flood-risk prediction
- **LayerCAM** for visual explanation of model predictions

The study covers **2014–2025** and focuses on **Satkhira, Khulna, and Bagerhat**.

---

## Research Gap

The project addresses several limitations identified in existing Sundarbans and Bangladesh flood-monitoring research:

1. **Optical imagery limitations**  
   Optical satellite imagery is strongly affected by cloud cover, particularly during monsoon conditions. SAR provides an all-weather sensing modality that can continue to provide observations under cloudy conditions.

2. **Single-modality analysis**  
   Image-only methods capture visual terrain and water-related structures but lack historical environmental context. Climate-only approaches provide numerical context but cannot directly observe spatial conditions.

3. **Limited multimodal integration**  
   The study identifies a lack of a consistent deep learning framework that jointly integrates SAR imagery with climate records for flood-risk assessment in the Sundarbans.

4. **Explainability limitations**  
   Flood-risk models may produce predictions without showing which physical regions contributed to those predictions. SARA-Net therefore incorporates LayerCAM to visualize spatial evidence.

5. **Complex hydrological behaviour**  
   Flood risk does not necessarily follow rainfall intensity alone. The study investigates **hydrological decoupling**, including high predicted risk under comparatively lower rainfall conditions.

---

## Objectives

The project was designed around four main objectives:

- Develop **SARA-Net**, a dual-stream model that fuses Sentinel-1 SAR imagery with climate information.
- Construct a multimodal dataset covering **2014–2025** for Satkhira, Khulna, and Bagerhat.
- Analyze long-term flood-risk patterns and identify regional vulnerability and possible hydrological decoupling.
- Apply **LayerCAM** to visualize the spatial features influencing flood-risk predictions.

---

## Key Contributions

The study presents four principal contributions:

1. **SARA-Net multimodal architecture** combining SAR imagery and climate data.
2. **All-weather monitoring formulation** based on Sentinel-1 SAR, addressing the limitations of cloud-dependent optical observations.
3. **Hydrological decoupling analysis** to investigate flood-risk patterns that are not explained by rainfall intensity alone.
4. **LayerCAM-based explainability** for identifying spatial regions associated with model predictions.

---

## Methodology

The complete SARA-Net pipeline follows four major stages:

<p align="center">
  <img src="figures/system-workflow.png" alt="SARA-Net system workflow" width="72%">
</p>

### 1. Data Acquisition

Two primary data sources are combined:

- **Sentinel-1 SAR imagery** from the *SAR image of Sunderbans* dataset
- **Climate records** from the *Bangladesh Climate Change Simulation Dataset*

The climate context includes:

- Annual rainfall (mm)
- River water level (m)

The SAR dataset provides three processing levels:

- **Enhanced Visualization** — RGB composites for qualitative analysis of major events
- **SAR Urban** — useful for infrastructure, embankments, and human encroachment
- **VH Polarization** — sensitive to vegetation structure and useful for mangrove monitoring

### 2. Automated Preprocessing

The visual and climate streams are synchronized before model training.

The preprocessing pipeline includes:

1. **OCR year extraction**  
   EasyOCR extracts year/timestamp information from SAR image frames.

2. **Ecological zone extraction**  
   Each image is divided into:
   - **North Zone:** approximately the upper 55%, representing human settlement areas
   - **South Zone:** approximately the remaining 45%, representing the mangrove forest shield

3. **District segmentation**  
   The image width is proportionally divided into:
   - Satkhira: 0–40%
   - Khulna: 40–70%
   - Bagerhat: 70–100%

4. **Multimodal synchronization**  
   Each processed SAR sample is matched with its corresponding rainfall, river-level, year, district, and sensor-mode context.

5. **Risk labeling**  
   Samples are classified as:
   - **High Risk (1):** flood-impact score above the dataset median
   - **Safe (0):** flood-impact score at or below the median

<p align="center">
  <img src="figures/preprocessed-data.png" alt="Preprocessed SAR samples" width="55%">
</p>

### 3. Data Augmentation

Training augmentation includes random spatial transformations and climate-feature noise injection. The final paper specifies random horizontal flips and affine translations for images, while 10% of climate features may be replaced with Gaussian noise during training. Validation and test data use only resizing and normalization.

### 4. Multimodal Learning

The processed inputs are passed through two parallel feature streams:

```text
                 SARA-Net
                    │
        ┌───────────┴───────────┐
        │                       │
   Visual Stream           Context Stream
   Pretrained              MLP
   ResNet-18
        │                       │
    512-dim                 32-dim
        │                       │
        └───────────┬───────────┘
                    │
             Concatenation
                 544-dim
                    │
             Dense 128 + ReLU
             + Dropout (0.3)
                    │
                Dense 64
                  + ReLU
                    │
              Sigmoid Output
                    │
             Risk Probability
```

<p align="center">
  <img src="figures/sara-net-architecture.png" alt="SARA-Net architecture" width="100%">
</p>

<p align="center">
  <img src="figures/sara-net-architecture-vertical.png" alt="Detailed SARA-Net architecture" width="48%">
</p>

---

## Model Architecture

### Visual Stream — ResNet-18

A pretrained **ResNet-18** backbone processes 224×224 SAR images. Its final classification layer is replaced by an identity mapping to obtain a **512-dimensional visual representation**, followed by dropout with \(p=0.5\).

The visual branch is intended to capture spatial patterns such as:

- Water structures
- Riverbanks
- Land boundaries
- Settlement edges
- Embankments
- Mangrove and tidal-channel structures

### Context Stream — MLP

The context branch receives a **5-dimensional environmental vector** containing normalized rainfall, normalized river level, and a one-hot sensor-mode representation.

The MLP follows:

```text
5 → 64 → 32
```

with Batch Normalization and ReLU activation.

### Fusion and Classification

The 512-dimensional visual representation and 32-dimensional context representation are concatenated:

```text
512 + 32 = 544 dimensions
```

The fusion head then follows:

```text
544 → 128 → 64 → 1
```

with ReLU activations and dropout in the fusion head. The final sigmoid output represents a continuous risk probability between **0 (Safe)** and **1 (High Risk)**.

The reported analysis indicates **SAR-dominant fusion behaviour**, with climate variables providing supporting contextual information.

---

## Dataset Description

The final study constructs a multimodal dataset containing **4,650 samples across 36 district-year groups**.

| Component | Description |
|---|---|
| Visual modality | Sentinel-1 SAR imagery |
| Climate modality | Rainfall + river-water level |
| Study period | 2014–2025 |
| Districts | Satkhira, Khulna, Bagerhat |
| SAR processing levels | Enhanced Visualization, SAR Urban, VH Polarization |
| Image input | 224 × 224 |
| Context vector | 5 features |
| Dataset groups | 36 district-year groups |
| Target | Synthetic mechanism-informed flood-impact target |

### Important Dataset Note

The benchmark uses a **synthetic, mechanism-informed flood-impact target**. The target is useful for evaluating the proposed modelling framework, but **it must not be interpreted as measured real-world flood impact**.

This distinction is important when reproducing or extending the experiments.

---

## Experimental Setup

The final paper reports:

- Framework: **PyTorch**
- Training environment: **NVIDIA T4 GPU via Google Colab**
- Train/validation/test split: **group-disjoint**
- Groups: **24 training + 6 validation + 6 test**
- Training epochs: **10**
- Batch size: **32**
- Loss: **Binary Cross-Entropy (BCE)**
- Optimizer: **Adam**
- Learning rate: **5 × 10⁻⁵**
- Weight decay: **1 × 10⁻⁴**
- Scheduler: **ReduceLROnPlateau**
- Additional evaluation: **Nested grouped cross-validation**

Group-disjoint splitting prevents samples belonging to the same district-year group from appearing across evaluation partitions.

---

## Results

### Fixed Group-Disjoint Test Set

SARA-Net achieves:

| Model | Modality | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Climate-Only (MLP) | Climate | 48.08% | 0.45 | 0.35 | 0.39 |
| Visual-Only (ResNet-18) | SAR Images | 76.34% | 0.71 | 0.88 | 0.78 |
| **SARA-Net** | **Fusion** | **76.98%** | **0.68** | **0.97** | **0.80** |

**SARA-Net ROC-AUC: 0.771**

### Backbone Comparison

| Backbone | Accuracy | AUC |
|---|---:|---:|
| ResNet-18 | 76.98% | 0.771 |
| ResNet-50 | 78.26% | 0.786 |
| DenseNet-121 | 76.21% | 0.770 |
| EfficientNet-B0 | 71.99% | 0.778 |

The ResNet-50 experiment achieved the highest reported accuracy and AUC among the tested backbones, while ResNet-18 provided a lighter alternative with competitive performance.

### Nested Grouped Cross-Validation

SARA-Net achieved:

- **Accuracy:** 74.80 ± 12.31%
- **AUC:** 0.688 ± 0.211

The variability across grouped folds highlights the importance of evaluating generalization across district-year groups rather than relying only on random sample-level splitting.

---

## Hydrological Decoupling

The study investigates the relationship between annual rainfall and predicted flood risk across 2014–2025.

A key observation is that predicted risk does not always increase with rainfall intensity. For example, the paper reports a relatively low predicted risk during the **2018** high-rainfall period despite rainfall reaching **4537 mm**, while a higher predicted risk appears in **2022** under comparatively lower rainfall.

The 2022 pattern is described as a **possible silent-flood pattern**, not as a confirmed real-world flood event. The paper discusses drainage failure and embankment weakness as possible mechanisms that could contribute to such behaviour.

<p align="center">
  <img src="figures/discovery.png" alt="SARA-Net discovery: rainfall and visual-risk decoupling" width="100%">
</p>

---

## Ecosystem Shield Analysis

The study introduces a **Hydrological Resilience Index (HRI)** to examine the flood-buffering capacity of the mangrove ecosystem.

The reported analysis shows positive HRI values during 2014–2020, followed by a decline after 2020 and particularly around the 2022 event.

This analysis is used to investigate changes in the ecosystem's buffering capacity alongside regional flood-risk patterns.

<p align="center">
  <img src="figures/ecosystem-shield.png" alt="Ecosystem shield and hydrological resilience analysis" width="100%">
</p>

---

## Regional Vulnerability

The regional analysis reports the following risk scores:

| Region | District | Risk Score |
|---|---|---:|
| Western | Satkhira | **0.88** |
| Central | Khulna | 0.65 |
| Eastern | Bagerhat | 0.42 |

The paper identifies Satkhira as having the highest reported regional risk score in this analysis.

<p align="center">
  <img src="figures/regional-vulnerability.png" alt="Regional vulnerability profile" width="75%">
</p>

---

## Explainable AI — LayerCAM

SARA-Net incorporates **LayerCAM** to investigate which spatial regions contribute to the model's predictions.

Forward and backward hooks are attached to the final convolutional layer of the ResNet-18 backbone. The captured activation maps and gradients are aggregated to produce spatial localization heatmaps.

The reported visualizations show activation around features including:

- Riverbanks
- Embankments
- Settlement boundaries
- Tidal channels
- Mangrove canopy structures

The North/settlement zone shows strong activation around linear and boundary structures, while the South/forest zone emphasizes canopy and tidal-channel structures.

<p align="center">
  <img src="figures/layercam-analysis.png" alt="LayerCAM visual explanations" width="90%">
</p>

The paper interprets these activations as evidence that the model is responding to meaningful environmental structures rather than only irrelevant image background.

---

## Research Outputs

This repository contains multiple development notebooks together with the project's research paper, report, and final presentation.

### Notebooks

| Notebook | Description |
|---|---|
| [`SARA-Net-v3.ipynb`](notebooks/SARA-Net-v3.ipynb) | Earlier SARA-Net development implementation |
| [`SARA-Net-v4.ipynb`](notebooks/SARA-Net-v4.ipynb) | Intermediate model development and evaluation |
| [`SARA-Net-v7.ipynb`](notebooks/SARA-Net-v7.ipynb) | Later experimental/integrity revision |

The notebooks are preserved as development artifacts. Reported paper metrics should be interpreted according to the experimental setup documented in the **latest paper**, rather than assumed to apply identically to every notebook version.

### Reports & Presentation

- [`SARA-Net-Paper.pdf`](reports/SARA-Net-Paper.pdf) — latest research paper
- [`SARA-Net-Report.pdf`](reports/SARA-Net-Report.pdf) — project report
- [`SARA-Net-Final-Presentation.pptx`](reports/SARA-Net-Final-Presentation.pptx) — final presentation

---

## Repository Structure

```text
SARA-Net-Sundarbans/
│
├── README.md
│
├── figures/
│   ├── discovery.png
│   ├── ecosystem-shield.png
│   ├── layercam-analysis.png
│   ├── preprocessed-data.png
│   ├── regional-vulnerability.png
│   ├── results-dashboard.png
│   ├── sara-net-architecture.png
│   ├── sara-net-architecture-vertical.png
│   └── system-workflow.png
│
├── notebooks/
│   ├── SARA-Net-v3.ipynb
│   ├── SARA-Net-v4.ipynb
│   └── SARA-Net-v7.ipynb
│
└── reports/
    ├── SARA-Net-Final-Presentation.pptx
    ├── SARA-Net-Paper.pdf
    └── SARA-Net-Report.pdf
```

---

## Technologies

- Python
- PyTorch
- TorchVision
- ResNet-18
- Multi-Layer Perceptron (MLP)
- Sentinel-1 SAR
- LayerCAM
- EasyOCR
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Google Colab / NVIDIA T4

---

## Data Sources

The study uses the following datasets cited by the paper:

- **SAR image of Sunderbans: A decade of Sentinel-1 SAR observations over the Sundarbans**  
  https://www.kaggle.com/datasets/sohambenji/sar-image-of-sunderbans

- **Bangladesh Climate Change Simulation Dataset**  
  https://www.kaggle.com/datasets/shohinurpervezshohan/bangladesh-climate-change-simulation-dataset

The repository does **not** redistribute the source datasets.

---

## Limitations and Responsible Interpretation

SARA-Net is a research prototype for flood-risk assessment rather than an operational emergency-warning system.

The most important limitation is that the benchmark target is **synthetic and mechanism-informed**, not measured flood-impact ground truth. Therefore, the reported metrics quantify performance on the constructed benchmark and should not be interpreted as direct evidence of real-world flood-prediction accuracy.

The paper also reports substantial group-level variability in nested grouped cross-validation.

Future work identified by the study includes:

- Developing a lightweight version for edge deployment
- Connecting the framework to live Sentinel-1 observations
- Extending evaluation to other vulnerable coastal delta regions
- Strengthening validation with measured flood-impact observations

---

## Citation

If you use this repository or build upon the SARA-Net framework, please cite:

```bibtex
@article{saranet2025,
  title   = {SARA-Net: Multimodal SAR and Climate Data Fusion for Flood Risk Assessment in the Sundarbans},
  author  = {Asmaul Hossain Akash and Dolon Akter Mim and Tazin Jannat Bushra and Tithi Karmakar},
  year    = {2025}
}
```

> **Note:** The repository's `reports/SARA-Net-Paper.pdf` contains the complete author list and final reference information.

---

## Acknowledgements

This project was conducted as part of the **Computer Science Applications and Advancements** course at the **American International University-Bangladesh (AIUB)**.

The project report acknowledges the guidance and support of **Dr. Muhammad Hasibur Rashid Chayon**, Department of Computer Science, AIUB.

---

## Authors

**Asmaul Hossain Akash**  
Department of Computer Science  
American International University-Bangladesh (AIUB)

**Dolon Akter Mim**  
Department of Computer Science  
American International University-Bangladesh (AIUB)

**Tazin Jannat Bushra**  
Department of Computer Science  
American International University-Bangladesh (AIUB)

**Tithi Karmakar**  
Department of Computer Science  
American International University-Bangladesh (AIUB)

**Supervisor:** Dr. Muhammad Hasibur Rashid Chayon

---

<p align="center">
  <i>SARA-Net — Exploring multimodal AI for resilient coastal ecosystems.</i>
</p>
