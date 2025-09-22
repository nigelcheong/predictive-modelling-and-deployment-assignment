# Predictive Modelling and Deployment  

This project demonstrates an end-to-end **machine learning workflow** for restaurant data, including:  

- **Data preprocessing**  
- **Feature engineering**  
- **Model training (regression + classification)**  
- **Experiment tracking and reproducibility with DVC**  
- **Data versioning with Git LFS**  

The workflow is fully reproducible using `dvc repro`, ensuring consistency across runs and portability across environments.  

---

## 📂 Repository Structure  

.
├── data/ # Raw and intermediate datasets (Git LFS tracked)
├── models/ # Trained models (DVC tracked, not stored in Git)
├── notebooks/ # Exploratory analysis & PySpark experiments
├── preprocess.py # Preprocessing script
├── feature_engineer.py # Feature engineering script
├── train.py # Training script (regression + classification)
├── dvc.yaml # DVC pipeline definition
├── dvc.lock # Pipeline lock file (snapshots of dependencies/outputs)
├── .gitignore # Ignores DVC outputs in Git
├── .dvcignore # Ignores temp/unwanted files in DVC
└── README.md # Project documentation

---

## ⚙️ Pipeline Overview  

The pipeline is defined in `dvc.yaml` and consists of three stages:  

1. **Preprocessing** (`preprocess`)  
   - Input: `data/zomato_df_final_data.csv`  
   - Script: `preprocess.py`  
   - Output: `data/preprocessed.csv`  

2. **Feature Engineering** (`feature_engineering`)  
   - Input: `data/preprocessed.csv`  
   - Script: `feature_engineer.py`  
   - Output: `data/engineered.csv`  

3. **Training** (`train_models`)  
   - Input: `data/engineered.csv`  
   - Script: `train.py`  
   - Outputs:  
     - `models/linear_regression.pkl`  
     - `models/svm_classifier.pkl`  
   - Also saves evaluation results in `results/`  

---

## 🚀 Usage  

### 1. Clone the Repository  
```bash
git clone <your-repo-url>
cd <your-repo>
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Pull Data and Models
```bash
git lfs pull
dvc pull
```
### 4. Reproduce the Pipeline
```bash
dvc repro
```
