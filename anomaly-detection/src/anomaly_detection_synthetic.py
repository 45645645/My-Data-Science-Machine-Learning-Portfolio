import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


def generate_synthetic_data(n_points: int = 500, n_anomalies: int = 20, random_state: int = 42):
    """
    產生一組簡單的時間序列資料：
    - 正常狀況：平滑的 sin 波 + 一點雜訊
    - 異常點：在隨機時間點上，數值被拉高或拉低
    """
    rng = np.random.RandomState(random_state)

    time_index = pd.date_range(start="2025-01-01", periods=n_points, freq="T")

    # baseline: 正常訊號（sin wave + noise）
    signal = np.sin(np.linspace(0, 10, n_points)) + rng.normal(0, 0.1, size=n_points)

    # 複製一份當作可被汙染的版本
    signal_with_anomaly = signal.copy()

    # 隨機挑幾個 index 當異常點
    anomaly_indices = rng.choice(n_points, size=n_anomalies, replace=False)
    # 在這些點大幅往上或往下偏移
    signal_with_anomaly[anomaly_indices] += rng.normal(0, 4.0, size=n_anomalies)

    df = pd.DataFrame(
        {
            "timestamp": time_index,
            "value": signal_with_anomaly,
        }
    )

    df["is_true_anomaly"] = 0
    df.loc[anomaly_indices, "is_true_anomaly"] = 1

    return df


def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42):
    """
    使用 Isolation Forest 做異常偵測。
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )

    # 只用數值欄位當特徵（這裡就用一維的 value）
    X = df[["value"]].values

    model.fit(X)
    # predict: -1 代表異常，1 代表正常
    pred = model.predict(X)
    scores = model.decision_function(X)

    df_result = df.copy()
    df_result["if_score"] = scores
    df_result["if_pred"] = (pred == -1).astype(int)  # 1 = predicted anomaly

    return df_result, model


def plot_results(df_result: pd.DataFrame, save_path: str | None = None):
    """
    把原始訊號 & 模型偵測到的異常點畫出來。
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(df_result["timestamp"], df_result["value"], label="signal", linewidth=1)

    # 畫出模型標記為異常的點
    anomalies = df_result[df_result["if_pred"] == 1]
    ax.scatter(
        anomalies["timestamp"],
        anomalies["value"],
        marker="o",
        s=30,
        label="predicted anomaly",
    )

    ax.set_title("Synthetic Time Series with Anomaly Detection (Isolation Forest)")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    plt.show()


def main():
    print("Generating synthetic data...")
    df = generate_synthetic_data()

    print("Running Isolation Forest...")
    df_result, model = run_isolation_forest(df)

    # 簡單列出偵測結果概況
    n_pred_anomaly = df_result["if_pred"].sum()
    n_true_anomaly = df_result["is_true_anomaly"].sum()

    print(f"True anomalies in data: {n_true_anomaly}")
    print(f"Predicted anomalies by model: {n_pred_anomaly}")

    # 儲存成 CSV，之後可以拿來做 notebook 分析
    df_result.to_csv("results/anomaly_detection_result.csv", index=False)

    # 畫圖
    plot_results(df_result, save_path="results/anomaly_detection_plot.png")


if __name__ == "__main__":
    # 確保有 results 資料夾
    import os

    os.makedirs("results", exist_ok=True)
    main()
