import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def load_data(path: str):
    """
    載入商品資料（可以是車子、手機、任何 tabular dataset）。
    為了展示流程，這裡將來可以放公開 dataset 或 synthetic data。
    """
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame):
    """
    - 只保留數值欄位（示範版）
    - 做 missing value 處理
    - 數值標準化
    回傳：(scaled array, scaler, feature_columns)
    """
    df_num = df.select_dtypes(include=[np.number]).copy()
    df_num = df_num.fillna(df_num.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)

    return X_scaled, scaler, df_num.columns.tolist()


def run_pca(X_scaled, n_components=2):
    """
    做 PCA 降維，把高維度資料壓成 2 或 3 維，方便可視化與推薦。
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


def build_knn_recommender(X_pca, n_neighbors=5):
    """
    在 PCA 空間建立 KNN 模型，用來根據距離找相似商品。
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X_pca)
    return knn


def recommend(knn, X_pca, query_index: int):
    """
    根據第 query_index 筆商品，找出最接近的幾個商品。
    """
    distances, indices = knn.kneighbors([X_pca[query_index]])
    return distances[0], indices[0]


def plot_pca_space(X_pca, query_index=None, neighbors=None):
    """
    把 PCA 空間畫出來：
    - 每個點是商品
    - query item 用紅色顯示
    - 推薦結果用黃色顯示
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, label="Items")

    if query_index is not None:
        plt.scatter(X_pca[query_index, 0], X_pca[query_index, 1],
                    color="red", s=80, label="Query Item")

    if neighbors is not None:
        plt.scatter(X_pca[neighbors, 0], X_pca[neighbors, 1],
                    color="gold", s=80, label="Recommended Items")

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("PCA Embedding of Items")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # ==== 1. 讀資料 ==== 
    df = pd.DataFrame({
        "price": [500, 520, 510, 900, 890, 880],
        "horsepower": [80, 82, 79, 150, 155, 148],
        "weight": [1100, 1120, 1080, 1500, 1520, 1480],
        "engine_size": [1.2, 1.3, 1.2, 2.0, 2.1, 1.9]
    })  
    # 這是一個示範用的 synthetic dataset
    # 之後可以換成車輛 dataset CSV

    # ==== 2. 前處理 ====
    X_scaled, scaler, features = preprocess_data(df)

    # ==== 3. PCA ====
    X_pca, pca = run_pca(X_scaled, n_components=2)

    # ==== 4. 建 KNN 模型 ====
    knn = build_knn_recommender(X_pca, n_neighbors=3)

    # ==== 5. 做推薦（挑第 0 筆商品來查相似商品）====
    query_index = 0
    distances, indices = recommend(knn, X_pca, query_index)

    print("Query item index:", query_index)
    print("Recommended items:", indices)
    print("Distances:", distances)

    # ==== 6. 畫 PCA 空間 ====
    plot_pca_space(X_pca, query_index=query_index, neighbors=indices)


if __name__ == "__main__":
    main()
