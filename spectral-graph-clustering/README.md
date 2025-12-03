# 🔮 Spectral Graph Clustering

This project demonstrates how **spectral graph theory** can be used to cluster graphs using the eigenvalues and eigenvectors of the graph Laplacian.

Inspired by my research background in **Cheeger cuts, λ₂ (the Fiedler value), and symmetric dumbbell graphs**, this project focuses on:

- Constructing synthetic graphs with clear community structure  
- Computing the graph Laplacian and its spectrum  
- Using eigenvectors (especially the Fiedler vector) for clustering  
- Visualizing the resulting clusters  

---

## 📌 Project Overview

Main goals:

- Show how the **Laplacian matrix** encodes connectivity  
- Use the **second smallest eigenvalue λ₂ and its eigenvector** for clustering  
- Compare simple thresholding vs. k-means on eigenvector embeddings  
- Visualize clusters in 2D using graph layouts  

This is a practical demonstration of ideas related to:

- Cheeger cuts  
- Spectral partitioning  
- Community detection on graphs  

---

## 📂 Folder Structure

Planned structure:

```text
spectral-graph-clustering/
 ├── data/                # Optional: graph data or adjacency matrices
 ├── notebooks/           # Jupyter notebooks for experiments
 ├── src/                 # Python scripts for spectral clustering
 ├── results/             # Plots of clustered graphs, eigenvalue distributions
 └── README.md            # Project documentation
