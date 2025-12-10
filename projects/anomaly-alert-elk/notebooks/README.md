# 🚀 Host Anomaly Early Warning — ELK/Kibana + Custom ES Client

以 ELK Logs 建立「**前兆預警**」並量化成效（Precision / Recall / F1 / **平均提前時間**）。  
支援三模式：**DEMO（合成）**、**LIVE_ES（連公司 ES）**、**UPLOAD（上傳四個 CSV）**。

<p align="left">
  <a href="https://colab.research.google.com/github/<YOUR_GH_USER>/adaline-ml-portfolio/blob/main/projects/anomaly-alert-elk/notebooks/Anomaly_Alert_Demo_Colab.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>
</p>

## 1) 問題定義（兩個月內要有可用產出）
- **目標**：提前告知可能的主機異常，供管理者預防性處置
- **評估**：能預警多少異常、可**提前多久**（mean lead time）、Precision/Recall/F1
- **限制**：無 Kibana rules 權限 → **自寫 ES Client + 批次告警**（Email/Webhook）

## 2) 方法概述
- **R1 規則**：15 分內 `S2="SOFT"` 次數 > 3（依設備分組）
- **Label**：`Status="red"`
- **時間型匹配**：`t_pred ≤ t_event ≤ t_pred + ϑ`（預設 30 分）
- **指標**：Precision / Recall / F1 / 平均提前時間（分）
- **產出**：  
  - `R1_eval_predictions.csv`（EqpName, t_pred, count）  
  - `R1_eval_labels.csv`（EqpName, t_event, cnt）  
  - `R1_eval_metrics_overall.csv`  
  - `R1_eval_metrics_per_eqp.csv`

## 3) 快速開始
- **Colab**：點上方徽章 → 依 Notebook 指示選擇 DEMO/LIVE_ES/UPLOAD
- **LIVE_ES 注意**：輸入 `ES_URL / Index / 認證（Basic or ApiKey）/ CA 憑證路徑`
- **隱私**：請勿將公司憑證/真實索引提交至 Git

## 4) 圖表 & 摘要
- 時間熱度圖（小時 × 15 分）
- 噪音設備（總預警 vs 60 分冷卻去重）
- 高價值設備（F1、平均提前時間）
- **一頁摘要**（Word/Markdown）：`docs/onepager_example.md`

## 5) 架構（示意）
ELK Logs → 查詢模組 → 規則模組（R1/R2…）→ 匹配評估 → 指標統計 → 通知（Email）→ 告警落庫/圖表

## 6) STAR
- **S**：需在 2 個月內建立「主機異常**預警**」能力，Kibana 無告警權限  
- **T**：設計可實作與可評估的前兆規則，並量化早知道時間  
- **A**：自寫 ES Client；R1=15m `SOFT`>3；以 `red` 作 label；時間型匹配出 TP/FP/FN；做 per-Eqp 分析降噪  
- **R**：輸出四個 CSV＋圖表＋一頁摘要；定位「噪音設備」與「高價值設備」，提出 **連續視窗 + 冷卻 60 分** 降噪策略

## 7) 後續 Roadmap
- R2：關鍵字前兆（例：proc/officecan/backup/重啟 事件）
- 1 分滑動視窗；連續 k 視窗成立；冷卻時間自適應
- 欄位對齊（EqpName/Host）；異常類型切面
