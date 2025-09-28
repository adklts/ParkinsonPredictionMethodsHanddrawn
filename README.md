# ParkinsonPredictionMethodsHanddrawn
Phd related work

This is the related work of my Phd, 

Having datasets between Healthy and Parkinsonian individuals regarding hand-drawn we are trying to predict the creator of the designs. 

Please use the data and the *.ipynb in order to test the models. 

Also the *.owl model is included in the work,

For further details clarifications please contact me. 

adklts@gmail.com


# ParkinsonPredictionMethodsHanddrawn

Comparative evaluation of three approaches for Parkinson’s disease detection from hand-drawn patterns (spirals, waves, handwriting features):  

- **Machine Learning (Random Forest)**  
- **Fuzzy Ontology-Based Reasoning (OWL, fuzzy logic)**  
- **Large Language Models (ChatGPT, zero-shot prompting)**  

This repository contains datasets, code, ontology models, and experimental outputs used in the study.

---

## 📂 Repository Structure

- `Datawave/`, `Spiral/`, `Waves/` → Datasets and experimental files  
- `plots/`, `figures/`, `boxplots/` → Visualization outputs (histograms, boxplots, membership functions)  
- `predictions/` → Classifier outputs (confusion matrices, CSVs with predictions)  
- `.ipynb` notebooks → Code for training and evaluation across the 3 datasets  
- `.owl` files → Fuzzy ontology models created with Protégé  

---

## 🚀 Usage

1. Open one of the Jupyter notebooks (`Spiral1.ipynb`, `Waves.ipynb`, etc.).  
2. Use pre-extracted features (`extracted_features*.csv`) for model input.  
3. Train and evaluate models:
   - **Random Forest** (trained on real + CTGAN synthetic data)  
   - **Fuzzy ontology** (Protégé `.owl` files, fuzzy rules, membership functions)  
   - **ChatGPT zero-shot classification** (structured prompts with handwriting features)  
4. Results (plots, confusion matrices, fuzzy membership plots) are saved in `/plots/`, `/figures/`, and `/boxplots/`.

---

## 📊 Datasets

- **Dataset 1:** Waves Images, **Synthetic dataset File:** ctgan_synthetic_1000_per_class.csv
- **Dataset 2:** Spiral Images, **Synthetic dataset File:** ctgan_true_positives_all1000.csv
- **Dataset 3:** Spiral Arithmetic data from Spiral Handdrawn, **Synthetic dataset File:** healthy_synthetic_data20250511_optimized.csv, parkinsonian_synthetic_data20250511_optimized.csv



---

## 📈 Results

- **Fuzzy ontology:** Achieved **F1 = 0.91** on Dataset 3 (stylus pressure, speed, and segment metrics).  
- **Random Forest:** Trained on CTGAN synthetic data, reached **F1 ≈ 0.70–0.80** across datasets.  
- **ChatGPT (zero-shot):** Reached **F1 = 0.50**, serving as an explainable but less accurate baseline.  

Representative visualizations (histograms, boxplots, fuzzy membership plots) are available in the repository.

---

## 🔄 Reproducibility

- All fuzzy rules and triangular membership functions are included.  
- Rule sets (`candidate_fuzzy_rules_from_none.csv`, `.owl`) are provided for Protégé.  
- Synthetic augmentation with CTGAN is included (`ctgan_synthetic_*.csv`).  
- True Positive synthetic subsets are saved in `ctgan_true_positives_*.csv`.  

---
