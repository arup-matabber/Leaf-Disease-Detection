# LeafDiseaseDetection

A Python-based **plant leaf disease detection and classification** project. This repository provides code to train and evaluate models, compare performance across them, and run a Flask web app to predict diseases from leaf images.

> Forked from AmitMandhana/LeafDiseaseDetection.

---

## 📌 Overview

This repository includes:

* **Model training & fine-tuning scripts**
* **Evaluation & comparison of models**
* **Flask web application** to upload leaf images and get disease predictions
* **Pre-trained models** stored in `model/` and `model_finetuned/`
* **Disease metadata** in `disease_data.json`
* Test script to verify performance

---

## 📁 Repository Structure

```
LeafDiseaseDetection/
├── model/                    # Pre-trained models
├── model_finetuned/          # Fine-tuned model weights
├── app.py                    # Flask application for inference
├── compare_models.py         # Compare different models
├── diag_model.py             # Model architecture & inference logic
├── evaluate_models.py        # Evaluate models
├── train_finetune.py         # Train or fine-tune models
├── test_retrieval.py         # Test classification/retrieval
├── disease_data.json         # Disease labels & metadata
├── requirements.txt          # Dependencies
└── .gitignore
```

---

## 🚀 Getting Started

### ⚙️ Prerequisites

Install Python (3.8 or later is recommended).

---

### 📥 Installation

Clone the repository:

```bash
git clone https://github.com/EnthusiastiCoder/LeafDiseaseDetection.git
cd LeafDiseaseDetection
```

Create and activate a virtual environment:

```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training

Train or fine-tune a model with:

```bash
python train_finetune.py
```

Configure dataset paths and hyperparameters inside the script before training.

---

## 📊 Evaluate Trained Models

To evaluate models on test/validation sets:

```bash
python evaluate_models.py
```

To compare performance across saved models:

```bash
python compare_models.py
```

---

## 🧪 Test Script

A helper script for testing/classification:

```bash
python test_retrieval.py
```

---

## 🌐 Run the Inference App

Start the Flask server:

```bash
python app.py
```

Open your browser and go to:

```
http://localhost:5000
```

Upload a leaf image to receive a **disease prediction** and **confidence score**.

---

## 🧾 Output

The app shows:

* Predicted disease category
* Confidence level
* Optional metadata from `disease_data.json`

---

## 🛠️ Customization

You can improve/extend the project by:

* Training on your own leaf disease dataset
* Adding more classification models
* Enhancing the web frontend UI
* Exporting the model for mobile/web deployment

---

## 📦 Dependencies

See `requirements.txt` for all Python packages used.

[1]: https://github.com/EnthusiastiCoder/LeafDiseaseDetection "GitHub - EnthusiastiCoder/LeafDiseaseDetection"
