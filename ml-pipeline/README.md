# ⚙️ ML Pipeline – PCA-based Recommendation System

This project demonstrates a complete **machine learning pipeline** inspired by an AI car recommendation system I worked on in a hackathon.

The idea is to:

- Take tabular product data (e.g. car specs or item features)  
- Clean and preprocess the data  
- Apply **PCA (Principal Component Analysis)** for dimensionality reduction  
- Use distances / k-nearest neighbors / clustering in the PCA space  
- Generate simple recommendation results for "similar items"  

---

## 📌 Project Goals

- Show an end-to-end ML workflow:
  1. Data loading  
  2. Cleaning & preprocessing  
  3. Feature scaling  
  4. PCA  
  5. Similarity / clustering based recommendation  
  6. Evaluation & visualization  

- Connect theory to practice:
  - PCA as a way to capture major variation in features  
  - Using the PCA space as a compact representation for recommendation  

---

## 📂 Folder Structure (Planned)

```text
ml-pipeline/
 ├── data/                    # Sample product/car dataset
 ├── notebooks/               # Jupyter notebooks for EDA & pipeline
 ├── src/                     # Reusable pipeline code (PCA + kNN, etc.)
 ├── results/                 # Plots, embeddings, and sample recommendations
 └── README.md                # Project documentation
