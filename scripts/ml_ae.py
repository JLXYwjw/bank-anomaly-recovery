# -*- coding: utf-8 -*-
"""银行自编码器(AE)异常检测消融 —— 用 torch 实测无监督 DL, 与 Isolation Forest 同口径对比。

训练细节(全部确定):
- 特征: amount_log, balance_log, balance_missing, hour, is_late_night, over_balance, age
        (与 ml_analysis.py 完全一致, 保证可比)
- 归一化: StandardScaler(全量拟合, 无监督下常规做法; 与 IF 同输入口径)
- 网络: 7→32→8(bottleneck)→32→7, 编码器与解码器均 ReLU + Dropout(0.2)
- 损失: 均方误差重建损失 MSE(y, x̂) —— AE 通过压缩学习正常模式的低维表示, 异常点难重建, 误差大
- 优化器: Adam(lr=1e-3, betas=(0.9,0.999))
- 训练: minibatch 256, 最大 60 epoch, 早停(监控验证重建损失, patience=10)
- 切分: 前80%训练, 后20%验证(验证只做早停监控, 不参与学习)
- 异常评分: 每样本重建误差(越大越异常), 取 Top1% 标记异常(与 IF 同阈值口径)

输出: ml_result_ae.json + bank_scored.parquet 追加 ae_score/ae_label 两列。
结论方向: 无监督下 AE 与 IF 同口径(Top1%), 用重叠率/金额占比衡量一致性;
          AE 不可解释(无法说明"为何这笔重建差、对应什么业务含义"), 故主方法仍为 IF(可解释)。
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent.parent / "data" / "output"
clean = pd.read_parquet(OUT / "bank_clean.parquet")

# 特征(与 ml_analysis.py 一致)
f = pd.DataFrame(index=clean.index)
f["amount_log"] = np.log1p(clean["amount"])
f["balance_log"] = np.log1p(clean["balance"].fillna(0))
f["balance_missing"] = clean["balance"].isna().astype(int)
f["hour"] = clean["hour"]
f["is_late_night"] = clean["is_late_night"].astype(int)
f["over_balance"] = clean["over_balance"].astype(int)
f["age"] = clean["age"].fillna(clean["age"].median())

scaler = StandardScaler()
X = scaler.fit_transform(f.values).astype(np.float32)
n = len(X)
idx = np.arange(n)
tr = idx[: int(n * 0.8)]
va = idx[int(n * 0.8):]
rng = np.random.RandomState(42)


class AE(nn.Module):
    def __init__(self, d, h1=32, h2=8, drop=0.2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(drop))
        self.dec = nn.Sequential(nn.Linear(h2, h1), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(h1, d))

    def forward(self, x):
        return self.dec(self.enc(x))


torch.manual_seed(42)
model = AE(X.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
lossf = nn.MSELoss()
xt = torch.from_numpy(X[tr])
xv = torch.from_numpy(X[va])
bs, epochs, patience = 256, 60, 10
best = float("inf")
best_sd = None
no_imp, best_epoch = 0, 0
for ep in range(epochs):
    model.train()
    perm = rng.permutation(len(tr))
    for i in range(0, len(tr), bs):
        xb = xt[perm[i:i + bs]]
        opt.zero_grad()
        rec = model(xb)
        loss = lossf(rec, xb)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        vloss = lossf(model(xv), xv).item()
    if vloss < best - 1e-4:
        best = vloss
        best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        best_epoch = ep + 1
        no_imp = 0
    else:
        no_imp += 1
        if no_imp >= patience:
            break
model.load_state_dict(best_sd)

model.eval()
with torch.no_grad():
    rec_all = model(torch.from_numpy(X)).numpy()
ae_err = ((rec_all - X) ** 2).mean(axis=1)
clean["ae_score"] = ae_err
clean["ae_label"] = (ae_err >= np.quantile(ae_err, 0.99)).astype(int)

r = {}
r["n"] = int(n)
r["ae_flagged"] = int(clean["ae_label"].sum())
r["ae_flagged_amt"] = round(float(clean.loc[clean["ae_label"] == 1, "amount"].sum()), 2)
r["ae_flagged_pct_of_total"] = round(float(
    clean.loc[clean["ae_label"] == 1, "amount"].sum() / clean["amount"].sum() * 100), 2)
r["ae_top1pct_amt_share"] = r["ae_flagged_pct_of_total"]
r["train_info"] = {"architecture": "7-32-8-32-7", "loss": "MSE(reconstruction)",
                   "optimizer": "Adam(lr=1e-3, betas=(0.9,0.999))", "dropout": 0.2,
                   "normalization": "StandardScaler(full fit)", "batch_size": bs,
                   "max_epochs": epochs, "early_stop_patience": patience,
                   "epochs_done": best_epoch, "best_val_recon_loss": round(float(best), 6),
                   "train_val_split": "80/20 time-ordered", "seed": 42}

# 与 IF 对比(IF 标签在 bank_scored.parquet, 由 ml_analysis.py 生成)
try:
    scored = pd.read_parquet(OUT / "bank_scored.parquet")
    merge = clean.merge(scored[["txn_id", "if_label"]], on="txn_id", how="left")
    overlap = int(((merge["ae_label"] == 1) & (merge["if_label"] == 1)).sum())
    r["overlap_ae_and_if"] = overlap
    r["ae_overlap_pct_of_ae"] = round(overlap / r["ae_flagged"] * 100, 2) if r["ae_flagged"] else None
    r["if_flagged_amt"] = round(float(merge.loc[merge["if_label"] == 1, "amount"].sum()), 2)
    r["if_flagged_pct_of_total"] = round(float(
        merge.loc[merge["if_label"] == 1, "amount"].sum() / merge["amount"].sum() * 100), 2)
except Exception as e:
    r["overlap_ae_and_if"] = None
    r["if_flagged_amt"] = None

# 更新 bank_scored 加 ae 两列
scored2 = pd.read_parquet(OUT / "bank_scored.parquet")
ae_cols = clean[["txn_id", "ae_score", "ae_label"]]
scored2 = scored2.merge(ae_cols, on="txn_id", how="left")
scored2.to_parquet(OUT / "bank_scored.parquet", index=False)

with open(OUT / "ml_result_ae.json", "w", encoding="utf-8") as f:
    json.dump(r, f, ensure_ascii=False, indent=1)
print(json.dumps(r, ensure_ascii=False), flush=True)
