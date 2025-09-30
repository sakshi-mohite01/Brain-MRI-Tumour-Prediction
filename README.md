# 👋 Brain MRI Tumour Prediction & Decision Support App

![Brain MRI Banner](https://user-images.githubusercontent.com/your-banner-link-here)

🎓 MSc Data Science Student | University of Greenwich, London  
💻 Aspiring Data Scientist / Data Analyst  
📊 Skilled in Python, SQL, PyTorch, scikit-learn, Power BI, Streamlit  

---

## 📖 Project Overview

The **Brain MRI Tumour Prediction & Decision Support App** is a deep learning-powered solution designed to assist medical professionals in detecting and segmenting brain tumours from MRI scans.  
By combining CNN-based classification, segmentation, explainability, and interactive reporting, this project delivers an intuitive and reliable tool for supporting medical diagnosis.

---

## 🚀 Features

- 🎯 **High-Accuracy Tumour Classification** – CNN model achieving **97% accuracy** for brain tumour detection.  
- 🧠 **Tumour Segmentation** – Automatically segments tumour regions and overlays them on MRI scans for enhanced visualization.  
- 🔍 **Grad-CAM Explainability** – Generates heatmaps to illustrate which regions influenced predictions.  
- 📄 **Automated PDF Report Generation** – Creates detailed reports summarizing results for medical records.  
- 💬 **Interactive Chatbot** – Provides explanations and guidance to users based on prediction results.  
- 🛠 **User-Friendly Streamlit App** – Offers an interactive interface for tumour prediction, segmentation, and report generation.




## 📂 Dataset

This project uses the **BRISC2025 Brain MRI Dataset**, sourced from Kaggle.  
The dataset contains MRI scans for brain tumour classification and segmentation tasks.

### Dataset Structure
The dataset contains two main folders:
- `classification_task` → MRI scans with labels for tumour classification.
- `segmentation_task` → MRI scans and corresponding masks for tumour segmentation.

### Download Link
You can download the dataset from Kaggle here:  
👉 [BRISC2025 Dataset on Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025)  

### Usage
After downloading:
1. Place the dataset folder in the root directory of this project.  
2. Rename it as `brisc2025/` so the code can access it easily.

## ⚙️ Installation

Follow these steps to set up and run the Brain MRI Tumour Prediction & Decision Support App locally.

### 1. Clone the Repository
If you haven’t already cloned the project repository, run:
```bash
git clone https://github.com/sakshi-mohite01/brain-mri-tumour-prediction.git
cd brain-mri-tumour-prediction

2. **Create & Activate Conda Environment**  
I have created a dedicated environment for this project called `brain_tumour`.

```bash
conda create --name brain_tumour python=3.10
conda activate brain_tumour

3. **Install Dependencies**
Install all required Python libraries:
pip install -r requirements.txt

4. **Download the Dataset**
🔹 Manual Method
brisc2025/

5. **Run the Streamlit App**
Make sure your environment is active, then start the app:
streamlit run streamlit_app.py

💡 Hint: Always activate your environment before running the project:
conda activate brain_tumour

## 🚀 Usage

After installing the dependencies and setting up the environment, you can run the application using Streamlit.

## 📂 Project Structure

brain-mri-tumour-prediction/
│
├── brisc2025/                  # Dataset folder (classification_task & segmentation_task)
│   ├── classification_task/
│   ├── segmentation_task/
│
├── models/                     # Saved model files
│   ├── tumour_classification_model.pth
│   └── segmentation_model.pth
│
├── outputs/                    # Generated outputs (reports, images, logs)
│   ├── reports/
│   └── segmented_images/
│
├── .gitignore                  # Specifies intentionally untracked files to ignore
├── requirements.txt            # List of Python dependencies
├── streamlit_app.py            # Streamlit application entry point
├── source_code.ipynb           # Jupyter Notebook with model development & analysis
└── README.md                   # Project documentation






