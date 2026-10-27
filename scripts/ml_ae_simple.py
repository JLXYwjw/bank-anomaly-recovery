# -*- coding: utf-8 -*-
"""
自编码器异常检测（消融实验）—— 基础可读版（ml_ae_simple.py）
=================================================================
这个文件做的事，和原版 ml_ae.py 完全一样：用 PyTorch 搭一个小型自编码器（AE），
在无监督条件下实测深度学习方法，并和孤立森林（IF）按同一口径对比。

训练细节（全部写死，保证可复现）：
  · 特征：和 ml_analysis.py 完全一致的 7 个特征（金额/余额对数、余额缺失、
          小时、深夜、超余额、年龄），保证两种方法可比
  · 归一化：StandardScaler 全量拟合（无监督下常规做法）
  · 网络：7 → 32 → 8（瓶颈）→ 32 → 7，编码器解码器都带 ReLU + Dropout(0.2)
  · 损失：重建误差 MSE —— AE 先把数据压到低维再还原；正常样本还原得好，
          异常样本还原误差大，误差大的就是异常
  · 优化器：Adam(lr=1e-3)，批大小 256，最多 60 轮，验证集 10 轮不进步就早停
  · 切分：前 80% 训练、后 20% 验证（验证只用来早停，不参与学习）
  · 评分：每笔交易的重建误差，取最大的 1% 标记为异常（和 IF 同阈值口径）

结论方向（原版就写明）：无监督下 AE 和 IF 同口径标记，但 AE 不可解释——
说不清楚"为什么这笔重建差、对应什么业务含义"，所以主方法仍然是可解释的 IF。

产物：
  · data/output/ml_result_ae.json            AE 侧统计 + 与 IF 的重叠
  · 在 bank_scored.parquet 末尾追加 ae_score / ae_label 两列

运行方式（和原版一样，没有命令行参数；要先跑过 ml_analysis 生成 bank_scored）：
    python scripts/ml_ae_simple.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# 0. 路径与读数据
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "output"
clean = pd.read_parquet(OUT / "bank_clean.parquet")

# ------------------------------------------------------------------
# 1. 构造特征表（和 ml_analysis.py 一模一样）
# ------------------------------------------------------------------
f = pd.DataFrame(index=clean.index)
f["amount_log"] = np.log1p(clean["amount"])
f["balance_log"] = np.log1p(clean["balance"].fillna(0))

bal_missing_list = []
for v in clean["balance"]:
    if pd.isna(v):
        bal_missing_list.append(1)
    else:
        bal_missing_list.append(0)
f["balance_missing"] = bal_missing_list

f["hour"] = clean["hour"]

late_int_list = []
for v in clean["is_late_night"]:
    if bool(v):
        late_int_list.append(1)
    else:
        late_int_list.append(0)
f["is_late_night"] = late_int_list

over_int_list = []
for v in clean["over_balance"]:
    if bool(v):
        over_int_list.append(1)
    else:
        over_int_list.append(0)
f["over_balance"] = over_int_list

f["age"] = clean["age"].fillna(clean["age"].median())

# ------------------------------------------------------------------
# 2. 标准化 + 划分训练/验证
# ------------------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(f.values).astype(np.float32)
n = len(X)

# 按行顺序切：前 80% 训练，后 20% 验证
tr = np.arange(int(n * 0.8))
va = np.arange(int(n * 0.8), n)

# 训练时打乱用的随机数发生器，种子固定 42
rng = np.random.RandomState(42)


# ------------------------------------------------------------------
# 3. 自编码器网络结构
# ------------------------------------------------------------------
class AutoEncoder(nn.Module):
    """7 → 32 → 8 → 32 → 7：先压缩再还原，中间的 8 维就是"瓶颈"。"""

    def __init__(self, d, h1=32, h2=8, drop=0.2):
        super().__init__()
        # 编码器：d → 32 → 8
        self.enc = nn.Sequential(
            nn.Linear(d, h1), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(drop),
        )
        # 解码器：8 → 32 → d
        self.dec = nn.Sequential(
            nn.Linear(h2, h1), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h1, d),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


# ------------------------------------------------------------------
# 4. 训练循环
# ------------------------------------------------------------------
torch.manual_seed(42)
model = AutoEncoder(X.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
lossf = nn.MSELoss()

xt = torch.from_numpy(X[tr])
xv = torch.from_numpy(X[va])

batch_size = 256
max_epochs = 60
patience = 10

best = float("inf")       # 目前最好的验证损失
best_sd = None            # 对应最好那一轮的模型参数
no_improve = 0            # 连续多少轮没进步
best_epoch = 0

for ep in range(max_epochs):
    # ---- 训练阶段：打乱后按批更新 ----
    model.train()
    perm = rng.permutation(len(tr))
    for i in range(0, len(tr), batch_size):
        xb = xt[perm[i:i + batch_size]]
        opt.zero_grad()
        rec = model(xb)
        loss = lossf(rec, xb)
        loss.backward()
        opt.step()

    # ---- 验证阶段：不算梯度，只看重建损失 ----
    model.eval()
    with torch.no_grad():
        vloss = lossf(model(xv), xv).item()

    # 比最好成绩再好一点点才认为有进步，否则计数
    if vloss < best - 1e-4:
        best = vloss
        best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        best_epoch = ep + 1
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            break

# 验证集表现最好的那轮参数，作为最终模型
model.load_state_dict(best_sd)

# ------------------------------------------------------------------
# 5. 给全量数据算重建误差，取前 1% 标记异常
# ------------------------------------------------------------------
model.eval()
with torch.no_grad():
    rec_all = model(torch.from_numpy(X)).numpy()

# 每笔交易：7 个特征各自的误差平方，平均 → 一个重建误差
ae_err = ((rec_all - X) ** 2).mean(axis=1)
clean["ae_score"] = ae_err

ae_cut = np.quantile(ae_err, 0.99)
ae_label_list = []
for v in ae_err:
    if v >= ae_cut:
        ae_label_list.append(1)
    else:
        ae_label_list.append(0)
clean["ae_label"] = ae_label_list

# ------------------------------------------------------------------
# 6. 汇总 AE 侧统计
# ------------------------------------------------------------------
r = {}
r["n"] = int(n)
r["ae_flagged"] = int(clean["ae_label"].sum())
r["ae_flagged_amt"] = round(float(clean.loc[clean["ae_label"] == 1, "amount"].sum()), 2)
r["ae_flagged_pct_of_total"] = round(
    float(clean.loc[clean["ae_label"] == 1, "amount"].sum() / clean["amount"].sum() * 100), 2
)
r["ae_top1pct_amt_share"] = r["ae_flagged_pct_of_total"]
r["train_info"] = {
    "architecture": "7-32-8-32-7",
    "loss": "MSE(reconstruction)",
    "optimizer": "Adam(lr=1e-3, betas=(0.9,0.999))",
    "dropout": 0.2,
    "normalization": "StandardScaler(full fit)",
    "batch_size": batch_size,
    "max_epochs": max_epochs,
    "early_stop_patience": patience,
    "epochs_done": best_epoch,
    "best_val_recon_loss": round(float(best), 6),
    "train_val_split": "80/20 time-ordered",
    "seed": 42,
}

# ------------------------------------------------------------------
# 7. 和孤立森林结果对比（IF 标签在 bank_scored.parquet 里）
# ------------------------------------------------------------------
try:
    scored = pd.read_parquet(OUT / "bank_scored.parquet")
    merge = clean.merge(scored[["txn_id", "if_label"]], on="txn_id", how="left")
    overlap = 0
    for a_v, i_v in zip(merge["ae_label"], merge["if_label"]):
        if a_v == 1 and i_v == 1:
            overlap += 1
    r["overlap_ae_and_if"] = overlap
    if r["ae_flagged"]:
        r["ae_overlap_pct_of_ae"] = round(overlap / r["ae_flagged"] * 100, 2)
    else:
        r["ae_overlap_pct_of_ae"] = None
    r["if_flagged_amt"] = round(float(merge.loc[merge["if_label"] == 1, "amount"].sum()), 2)
    r["if_flagged_pct_of_total"] = round(
        float(merge.loc[merge["if_label"] == 1, "amount"].sum() / merge["amount"].sum() * 100), 2
    )
except Exception:
    # IF 产物还没生成就算了，不阻塞 AE 本身的结果
    r["overlap_ae_and_if"] = None
    r["if_flagged_amt"] = None

# ------------------------------------------------------------------
# 8. 把 AE 的两列追加进 bank_scored.parquet
# ------------------------------------------------------------------
scored2 = pd.read_parquet(OUT / "bank_scored.parquet")
ae_cols = clean[["txn_id", "ae_score", "ae_label"]]
scored2 = scored2.merge(ae_cols, on="txn_id", how="left")
scored2.to_parquet(OUT / "bank_scored.parquet", index=False)

# ------------------------------------------------------------------
# 9. 写报告并打印
# ------------------------------------------------------------------
with open(OUT / "ml_result_ae.json", "w", encoding="utf-8") as f:
    json.dump(r, f, ensure_ascii=False, indent=1)

print(json.dumps(r, ensure_ascii=False), flush=True)
