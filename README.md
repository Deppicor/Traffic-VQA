# Traffic-VQA: A Large-Scale Multimodal UAV Dataset for Advancing Cognitive Visual Question Answering in Complex Traffic Environments

[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/YuYu2004/Traffic-VQA)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
<!-- Optional: Add a link to your paper if available: [![Paper](https://img.shields.io/badge/paper-arXiv-B31B1B.svg)](https://arxiv.org/abs/YOUR_PAPER_ID) -->
<!-- Optional: Add a project page link: [![Project Page](https://img.shields.io/badge/Project-Page-Green.svg)](YOUR_PROJECT_PAGE_LINK) -->

> **TL;DR:** We introduce Traffic-VQA, the first large-scale, multimodal (optical-thermal) UAV dataset designed to benchmark cognitive Visual Question Answering (VQA) through complex reasoning in diverse, all-weather traffic scenarios.

---

<!-- **Traffic-VQA** addresses the growing need for intelligent systems capable of understanding and reasoning about dynamic traffic environments. This dataset provides a rich resource for developing and evaluating VQA models that can go beyond simple pattern recognition to perform complex cognitive tasks, such as inferring causality, predicting outcomes, and understanding nuanced interactions within traffic scenes captured by Unmanned Aerial Vehicles (UAVs). -->

Unmanned Aerial Vehicles (UAVs) provide crucial aerial perspectives for traffic surveillance. However, deriving crucial cognitive insights from UAV imagery, like identifying traffic violations and safety risks using Visual Question Answering (VQA), is limited by existing datasets, which often lack the required diversity, cognitive depth, and all-weather multimodal capabilities. To bridge this gap, we introduce **Traffic-VQA**, a large-scale benchmark of 8,180 aligned optical-thermal UAV image pairs and 1.3 million question-answer pairs, with an average of approximately 159 questions per image pair. This dataset encompasses 31 distinct question types, including 10 question types dedicated to complex cognitive reasoning and 15 types focused on fundamental perception, all specifically engineered to evaluate and advance understanding in challenging traffic scenarios including nighttime and adverse weather. The proposed Traffic-VQA is constructed using a novel human-LLM semi-automated annotation pipeline, ensuring both scale and cognitively rich content. It serves as a robust platform for benchmarking diverse Multimodal Large Language Models (MLLMs) and VQA models. To effectively assess the diverse and open-ended answers, we propose L3-Lite, a lightweight LLM-based metric for effective free-form QA evaluation. Extensive benchmarks demonstrate Traffic-VQA's utility in assessing cognitive capacities for complex traffic, highlighting multimodal robustness and revealing higher-order reasoning limitations in current models. The proposed dataset provides a valuable resource for developing and evaluating the next generation of all-weather, cognitive traffic monitoring systems powered by advanced visual understanding.

## 🌟 Key Features

*   **Large-Scale:** Comprises a significant number of question-answer pairs, images, and videos.
*   **Multimodal:** Includes both optical (RGB) and thermal imaging data, crucial for all-weather perception.
*   **UAV Perspective:** Offers an aerial viewpoint, common in modern traffic surveillance and management.
*   **Cognitive VQA:** Questions designed to elicit complex reasoning, moving beyond simple object recognition or attribute identification.
*   **Complex Traffic Environments:** Features diverse and challenging scenarios, including varying weather conditions, lighting, and traffic densities.
*   **Detailed Annotations:** Provides high-quality annotations to support various VQA tasks.

## 📢 News

*   **[May, 2025]** Traffic-VQA dataset and annotations are now open-sourced on [🤗 Hugging Face Datasets](https://huggingface.co/datasets/YuYu2004/Traffic-VQA)!

## 📊 Dataset Overview

*(Consider adding a small, compelling image or GIF here showing an example from your dataset: e.g., a split view of optical/thermal with a sample question)*

**Example:**
*   **Image:** (Description of an image with optical and thermal views)
*   **Question:** "If the red car continues at its current speed, will it likely need to brake for the pedestrian crossing the street in the thermal view?"
*   **Answer:** "Yes, the pedestrian is not clearly visible in the optical view due to poor lighting, but the thermal signature indicates their presence and trajectory, suggesting the car will need to brake."

(Add more details about dataset splits - train/val/test, number of images, QAs, etc. if you wish)

## 🚀 Getting Started

1.  **Download the Dataset:**
    Access the full Traffic-VQA dataset from [Hugging Face Datasets](https://huggingface.co/datasets/YuYu2004/Traffic-VQA).
2.  **Explore the Code:**
    This repository contains:
    *   Annotation tools and scripts.
    *   Evaluation code (`evaluation.py`) to benchmark models on Traffic-VQA.
    *   The L3-Lite evaluation metric implementation (`L3_Lite.py`).

    To run the evaluation:
    ```bash
    python evaluation.py --model_names <your_model_name> --result_path <path_to_your_model_results.json> --device <cuda_device_id>
    ```
    Ensure your `result.json` file is formatted with "question", "pred", and "gt" keys for each sample.

## ✔️ Baselines & Evaluation

We provide baseline results for several multimodal models evaluated on **Traffic-VQA**:

*   [LHRS-Bot](https://github.com/NJU-LHRS/LHRS-Bot)
*   [GeoChat](https://huggingface.co/MBZUAI/geochat-7B)
*   [Falcon](https://huggingface.co/TianHuiLab/Falcon-Single-Instruction-Large)
*   [MiniCPM-V](https://huggingface.co/openbmb/MiniCPM-V)
*   [MiniGPT-v2](https://huggingface.co/spaces/Vision-CAIR/MiniGPT-v2)
*   [DeepSeek-VL](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat)
*   [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
*   [RSAdapter](https://github.com/Y-D-Wang/RSAdapter)

The model we use for L3-Lite is [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

The evaluation utilizes our `L3-Lite` metric to assess semantic similarity between predicted answers and ground truths. Refer to `evaluation.py` for usage.

<!-- ## ✏️ Citation

If you use Traffic-VQA or the associated code in your research, please cite our work:

```bibtex
@misc{zhang2024trafficvqa,
      title={Traffic-VQA: A Large-Scale Multimodal UAV Dataset for Advancing Cognitive Visual Question Answering in Complex Traffic Environments},
      author={Yu Zhang and Your Other Co-authors},
      year={2024},
      eprint={Your_arXiv_ID_if_available},
      archivePrefix={arXiv},
      primaryClass={cs.CV} # or relevant category
} -->
