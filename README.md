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

.<br>
├── data/ # Raw and intermediate datasets (Git LFS tracked)<br>
├── models/ # Trained models (DVC tracked, not stored in Git)<br>
├── notebooks/ # Exploratory analysis & PySpark experiments<br>
├── preprocess.py # Preprocessing script<br>
├── feature_engineer.py # Feature engineering script<br>
├── train.py # Training script (regression + classification)<br>
├── dvc.yaml # DVC pipeline definition<br>
├── dvc.lock # Pipeline lock file (snapshots of dependencies/outputs)<br>
├── .gitignore # Ignores DVC outputs in Git<br>
├── .dvcignore # Ignores temp/unwanted files in DVC<br>
└── README.md # Project documentation<br>

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
git clone https://github.com/nigelcheong/predictive-modelling-and-deployment-assignment
cd predictive-modelling-and-deployment-assignment
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
