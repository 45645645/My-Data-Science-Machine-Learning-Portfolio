import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def build_two_cluster_graph(n1: int = 20, n2: int = 20, p_in: float = 0.4, p_out: float = 0.02, seed: int = 42):
    """
    建一個簡單的「兩群社群」隨機圖：
    - 兩個群體內部連線機率較高 (p_in)
    - 群體之間連線機率較低 (p_out)
    類似最基本的 stochastic block model。
    """
    rng = np.random.RandomState(seed)

    # 先建立兩塊 complete-like 的 adjacency matrix
    size = n1 + n2
    A = np.zeros((size, size), dtype=int)

    # 群體1內部
    for i in range(n1):
        for j in range(i + 1, n1):
            if rng.rand() < p_in:
                A[i, j] = A[j, i] = 1

    # 群體2內部
    for i in range(n1, n1 + n2):
        for j in range(i + 1, n1 + n2):
            if rng.rand() < p_in:
                A[i, j] = A[j, i] = 1

    # 兩群之間
    for i in range(n1):
        for j in range(n1, n1 + n2):
            if rng.rand() < p_out:
                A[i, j] = A[j, i] = 1

    G = nx.from_numpy_array(A)
    return G, A


def spectral_clustering(A: np.ndarray, k: int = 2, use_normalized: bool = False, random_state: int = 42):
    """
    對給定 adjacency matrix 做最基本的 spectral clustering：
    1. 建 Laplacian L = D - A 或 normalized Laplacian
    2. 取前 k 個最小特徵值對應的特徵向量
    3. 對這些特徵向量做 k-means 分群
    """
    # degree matrix
    degrees = A.sum(axis=1)
    D = np.diag(degrees)

    if not use_normalized:
        L = D - A
    else:
        # normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.where(degrees > 0, degrees, 1)))
        L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

    # 取最小的 k 個 eigenvalues 的 eigenvectors
    # eigh 回傳已排序特徵值
    eigvals, eigvecs = np.linalg.eigh(L)

    # 取前 k 個
    X_spec = eigvecs[:, :k]

    # 用 k-means 在 spectral embedding 上分群
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X_spec)

    return labels, eigvals


def plot_graph_with_clusters(G: nx.Graph, labels: np.ndarray, title: str = "Spectral Clustering Result"):
    """
    把圖畫出來，節點顏色依 cluster labels。
    """
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_color=labels, cmap="viridis", node_size=80)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    print("Building two-cluster synthetic graph...")
    G, A = build_two_cluster_graph()

    print("Running spectral clustering...")
    labels, eigvals = spectral_clustering(A, k=2, use_normalized=False)

    print("First 10 eigenvalues of L:")
    print(eigvals[:10])

    print("Plotting result...")
    plot_graph_with_clusters(G, labels, title="Spectral Clustering on Synthetic Two-Cluster Graph")


if __name__ == "__main__":
    main()
