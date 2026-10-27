# -*- coding: utf-8 -*-
r"""
报表构建 —— 基础可读版（build_bi_simple.py）
=============================================
把评分结果加工成
Tableau + PowerBI 双份报表要用的 CSV 和工程文件。

数据背景：
  · 1,048,567 笔 / 884,265 客户 / 9,355 个城市 —— 城市是长尾，必须 Top12 + 其他
  · 83.8% 的客户只有 1 笔交易，客户维度的"频次分析"基本没信息量，
    所以第 4 张图直接画"客户只交易了几笔"的分布，而不是客户排行
  · 金额极度右偏：<100 元占 19.0% 笔数却只 0.6% 金额；>20K 只占 1.3% 笔数却是 28.3% 金额
  · 风险队列按 if_score 排，不是按金额排 —— 队列里金额只有几十元也正常（洗钱特征是小额试探）

运行方式（没有命令行参数；要先跑过 ml_analysis）：
    python scripts/build_bi_simple.py


English:
Report building (plain, readable version).

Turns the scoring results into the CSV and project files needed by the
Tableau + Power BI twin reports.

Data context:
  - 1,048,567 transactions / 884,265 customers / 9,355 cities - cities are
    a long tail, so Top-12 + others is mandatory
  - 83.8% of customers have exactly 1 transaction, so a customer-level
    "frequency analysis" carries almost no information; chart 4 therefore
    draws the distribution of "how many transactions a customer made"
    instead of a customer ranking
  - Amount is heavily right-skewed: < 100 accounts for 19.0% of
    transactions but only 0.6% of amount; > 20K accounts for 1.3% of
    transactions but 28.3% of amount
  - The risk queue is ranked by if_score, not by amount - a queue member
    with only a few dozen yuan is normal (money-laundering patterns are
    small trial transfers)

Run (no command-line arguments; run ml_analysis first):
    python scripts/build_bi_simple.py
"""

import os
import sys

import pandas as pd

# ------------------------------------------------------------------
# 0. 路径配置
# ------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 引入同级项目的报表工具包（csvkit / pbipkit / twbkit）
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "tools", "bi_kit"))
import csvkit                      # noqa: E402
import pbipkit as K                # noqa: E402
import twbkit                      # noqa: E402

SRC = os.path.join(ROOT, "data", "output", "bank_scored.parquet")
QUE = os.path.join(ROOT, "data", "output", "queue_top50.parquet")
BI = os.path.join(ROOT, "data", "bi")

# 报表上要用的分类标签（顺序就是图上的展示顺序）
AMT_LAB = ["<100", "100-500", "500-1K", "1K-5K", "5K-20K", "20K-100K", ">100K"]
CUST_LAB = ["1 笔", "2 笔", "3-5 笔", "6-10 笔", "11-50 笔", ">50 笔"]
FLAG_ORDER = ["正常", "超额", "深夜", "两者都占"]


# ------------------------------------------------------------------
# 1. 数据准备：把明细加工成事实表 fact、队列表 q、汇总指标 f
# ------------------------------------------------------------------
def prep():
    d = pd.read_parquet(SRC)
    d["amount_m"] = d["amount"] / 1e6        # 金额换算成"百万"单位，图上好读
    total = d["amount"].sum()

    # ---- 1.1 金额分桶：<100 / 100-500 / ... / >100K ----
    edges = [0, 100, 500, 1000, 5000, 20000, 100000, 10 ** 9]
    d["amt_bucket"] = pd.cut(d["amount"], edges, labels=AMT_LAB, right=False).astype(str)

    # 每个桶的序号（画条形图时按这个顺序排，而不是按字母）
    amt_order_map = {}
    for i, v in enumerate(AMT_LAB):
        amt_order_map[v] = i
    d["amt_order"] = d["amt_bucket"].map(amt_order_map)

    # ---- 1.2 城市分组：交易数前 12 的城市保留原名，其余合并成"其他 N 个城市" ----
    c = d.groupby("city").agg(n=("txn_id", "size"), amt=("amount", "sum"))
    c = c.sort_values("n", ascending=False)
    top12 = list(c.head(12).index)

    # 标签里不放千分位逗号：CSV 里虽有引号保护，但能省一层麻烦就省
    other_label = "其他 " + str(len(c) - 12) + " 个城市"
    city_group_list = []
    for city in d["city"]:
        if city in top12:
            city_group_list.append(city)
        else:
            city_group_list.append(other_label)
    d["city_group"] = city_group_list

    # ---- 1.3 客户交易笔数分桶 ----
    cnt = d.groupby("customer_id").size()       # 每个客户交易了几笔
    cust_counts = d["customer_id"].map(cnt)    # 每笔交易所属客户的笔数
    cust_edges = [0, 1, 2, 5, 10, 50, 10 ** 6]
    d["cust_bucket"] = pd.cut(cust_counts, cust_edges, labels=CUST_LAB).astype(str)

    cust_order_map = {}
    for i, v in enumerate(CUST_LAB):
        cust_order_map[v] = i
    d["cust_order"] = d["cust_bucket"].map(cust_order_map)

    # ---- 1.4 异常类型四分类：正常 / 超额 / 深夜 / 两者都占 ----
    flag_list = []
    for ob_v, ln_v in zip(d["over_balance"], d["is_late_night"]):
        if bool(ob_v) and bool(ln_v):
            flag_list.append("两者都占")
        elif bool(ob_v):
            flag_list.append("超额")
        elif bool(ln_v):
            flag_list.append("深夜")
        else:
            flag_list.append("正常")
    d["flag_type"] = flag_list

    flag_order_map = {}
    for i, v in enumerate(FLAG_ORDER):
        flag_order_map[v] = i
    d["flag_order"] = d["flag_type"].map(flag_order_map)

    # ---- 1.5 拼事实表（喂给 Tableau / PowerBI 的明细表） ----
    fact = d[["hour", "city_group", "amt_bucket", "amt_order", "cust_bucket",
              "cust_order", "flag_type", "flag_order", "amount_m",
              "if_label"]].copy()
    fact["txn"] = 1                            # 每一行就是 1 笔交易，方便直接求和

    # 两个 0/1 标志列，改名后接在事实表末尾
    over_flag_list = []
    for v in d["over_balance"]:
        if bool(v):
            over_flag_list.append(1)
        else:
            over_flag_list.append(0)
    fact["over_flag"] = over_flag_list

    late_flag_list = []
    for v in d["is_late_night"]:
        if bool(v):
            late_flag_list.append(1)
        else:
            late_flag_list.append(0)
    fact["late_flag"] = late_flag_list

    # ---- 1.6 风险队列表：按异常分取前 20 笔，加个排名 ----
    # 原始 queue_top50 本身已经按 if_score 排好，这里再取前 20 并补 1~20 的名次
    q = pd.read_parquet(QUE)
    q = q.sort_values("if_score", ascending=False).head(20)
    q = q.reset_index(drop=True)

    rank_list = []
    for i in range(len(q)):
        rank_list.append(i + 1)
    q["rank"] = rank_list

    # 先挑出不要改的列，再依次追加换算后的金额和两个 0/1 标志列
    q = q[["rank", "customer_id", "city", "hour", "if_score"]]

    # 原始金额列不在上面的选择里，重新读一份前 20 笔来换算
    qraw = pd.read_parquet(QUE).sort_values("if_score", ascending=False).head(20)

    amount_m_list = []
    for v in qraw["amount"]:
        amount_m_list.append(v / 1e6)
    q["amount_m"] = amount_m_list

    q_late = []
    for v in qraw["is_late_night"]:
        if bool(v):
            q_late.append(1)
        else:
            q_late.append(0)
    q["late_flag"] = q_late

    q_over = []
    for v in qraw["over_balance"]:
        if bool(v):
            q_over.append(1)
        else:
            q_over.append(0)
    q["over_flag"] = q_over

    # ---- 1.7 汇总指标 f（打印日志 + 写进报表标题用） ----
    g = d.groupby("amt_bucket", observed=True).agg(
        n=("txn_id", "size"), amt=("amount", "sum")
    )
    big = g.loc[[AMT_LAB[4], AMT_LAB[5], AMT_LAB[6]]]   # 5K 以上三个桶
    h = d.groupby("hour").size()

    f = {}
    f["rows"] = len(d)
    f["cust"] = d["customer_id"].nunique()
    f["cities"] = len(c)
    f["amount"] = total / 1e9
    f["hour_peak"] = h.idxmax()
    f["hour_peak_pct"] = h.max() / len(d) * 100
    f["hour_low"] = h.idxmin()
    f["hour_low_n"] = int(h.min())
    f["city_top"] = c.index[0]
    f["city_top_pct"] = c["n"].iloc[0] / len(d) * 100
    f["city_top12"] = c["n"].head(12).sum() / len(d) * 100
    f["big_n_pct"] = big["n"].sum() / len(d) * 100
    f["big_a_pct"] = big["amt"].sum() / total * 100
    f["small_a_pct"] = g.loc[AMT_LAB[0], "amt"] / total * 100
    f["over_pct"] = d["over_balance"].mean() * 100
    f["over_amt"] = d["amount"][d["over_balance"]].sum() / 1e6
    f["late_pct"] = d["is_late_night"].mean() * 100
    f["late_amt"] = d["amount"][d["is_late_night"]].sum() / 1e6
    f["single_pct"] = (cnt == 1).mean() * 100
    f["if_pct"] = d["if_label"].mean() * 100
    f["if_amt_pct"] = d["amount"][d["if_label"] == 1].sum() / total * 100
    q_all = pd.read_parquet(QUE)
    f["q_all_late"] = int(q_all["is_late_night"].sum())
    f["q_all_over"] = int(q_all["over_balance"].sum())

    return fact, q, f


# ------------------------------------------------------------------
# 2. Tableau 工作簿（字段、图表、标题都是声明式配置）
# ------------------------------------------------------------------
def build_tableau(fact, q, f):
    F_DS, F_TS, F_CAP = "federated.0bank", "textscan.0bank", "交易明细"
    Q_DS, Q_TS, Q_CAP = "federated.0queue", "textscan.0queue", "风险队列"

    fact_fields = [
        ("hour",        "integer", "dimension", "ordinal",       20, "小时"),
        ("city_group",  "string",  "dimension", "nominal",      129, "城市分组"),
        ("amt_bucket",  "string",  "dimension", "nominal",      129, "金额区间"),
        ("amt_order",   "integer", "measure",   "quantitative",  20, "金额序"),
        ("cust_bucket", "string",  "dimension", "nominal",      129, "客户交易笔数"),
        ("cust_order",  "integer", "measure",   "quantitative",  20, "笔数序"),
        ("flag_type",   "string",  "dimension", "nominal",      129, "异常类型"),
        ("flag_order",  "integer", "measure",   "quantitative",  20, "类型序"),
        ("amount_m",    "real",    "measure",   "quantitative",   5, "金额(百万)"),
        ("if_label",    "integer", "measure",   "quantitative",  20, "IF 标记"),
        ("over_flag",   "integer", "measure",   "quantitative",  20, "超额标记"),
        ("late_flag",   "integer", "measure",   "quantitative",  20, "深夜标记"),
        ("txn",         "integer", "measure",   "quantitative",  20, "笔数"),
    ]
    q_fields = [
        ("rank",        "integer", "dimension", "ordinal",       20, "排名"),
        ("customer_id", "string",  "dimension", "nominal",      129, "客户"),
        ("city",        "string",  "dimension", "nominal",      129, "城市"),
        ("hour",        "integer", "dimension", "ordinal",       20, "小时"),
        ("amount_m",    "real",    "measure",   "quantitative",   5, "金额(百万)"),
        ("over_flag",   "integer", "measure",   "quantitative",  20, "超额"),
        ("late_flag",   "integer", "measure",   "quantitative",  20, "深夜"),
        ("if_score",    "real",    "measure",   "quantitative",   5, "异常分"),
    ]

    F, Q = F_DS, Q_DS
    N = "[" + F + "].[sum:txn:qk]"

    sheets = [
        dict(name="1 金额分布", ds=F,
             deps=[("string", "amt_bucket", "dimension", "nominal", "none:amt_bucket:nk", "None", "nominal"),
                   ("integer", "amt_order", "measure", "quantitative", "sum:amt_order:qk", "Sum", "quantitative"),
                   ("integer", "txn", "measure", "quantitative", "sum:txn:qk", "Sum", "quantitative")],
             rows="[" + F + "].[none:amt_bucket:nk]", cols=N, mark="Bar",
             encodings=[("text", N)], labels=True,
             sort=("[" + F + "].[none:amt_bucket:nk]", "[" + F + "].[sum:amt_order:qk]", "ASC"),
             num_format=(N, "#,##0"),
             title="%.1f%% 大额占 %.1f%% 金额" % (f["big_n_pct"], f["big_a_pct"])),

        dict(name="2 时段", ds=F,
             deps=[("integer", "hour", "dimension", "ordinal", "none:hour:ok", "None", "ordinal"),
                   ("integer", "txn", "measure", "quantitative", "sum:txn:qk", "Sum", "quantitative")],
             rows=N, cols="[" + F + "].[none:hour:ok]", mark="Bar",
             sort=("[" + F + "].[none:hour:ok]", N, "ASC"),
             num_format=(N, "#,##0"),
             title="%s 点最忙，占 %.1f%%" % (f["hour_peak"], f["hour_peak_pct"])),

        dict(name="3 城市", ds=F,
             deps=[("string", "city_group", "dimension", "nominal", "none:city_group:nk", "None", "nominal"),
                   ("integer", "txn", "measure", "quantitative", "sum:txn:qk", "Sum", "quantitative")],
             rows="[" + F + "].[none:city_group:nk]", cols=N, mark="Bar",
             encodings=[("text", N)], labels=True,
             sort=("[" + F + "].[none:city_group:nk]", N, "DESC"),
             num_format=(N, "#,##0"),
             title="%s 占 %.1f%%" % (f["city_top"], f["city_top_pct"])),

        dict(name="4 客户笔数", ds=F,
             deps=[("string", "cust_bucket", "dimension", "nominal", "none:cust_bucket:nk", "None", "nominal"),
                   ("integer", "cust_order", "measure", "quantitative", "sum:cust_order:qk", "Sum", "quantitative"),
                   ("integer", "txn", "measure", "quantitative", "sum:txn:qk", "Sum", "quantitative")],
             rows=N, cols="[" + F + "].[none:cust_bucket:nk]", mark="Bar",
             encodings=[("text", N)], labels=True,
             sort=("[" + F + "].[none:cust_bucket:nk]", "[" + F + "].[sum:cust_order:qk]", "ASC"),
             num_format=(N, "#,##0"),
             title="%.1f%% 客户只有 1 笔" % f["single_pct"]),

        dict(name="5 异常类型", ds=F,
             deps=[("string", "flag_type", "dimension", "nominal", "none:flag_type:nk", "None", "nominal"),
                   ("integer", "flag_order", "measure", "quantitative", "sum:flag_order:qk", "Sum", "quantitative"),
                   ("integer", "txn", "measure", "quantitative", "sum:txn:qk", "Sum", "quantitative")],
             rows=N, cols="[" + F + "].[none:flag_type:nk]", mark="Bar",
             encodings=[("text", N)], labels=True,
             sort=("[" + F + "].[none:flag_type:nk]", "[" + F + "].[sum:flag_order:qk]", "ASC"),
             num_format=(N, "#,##0"),
             title="超额 %.1f%% / 深夜 %.1f%%" % (f["over_pct"], f["late_pct"])),

        dict(name="6 风险队列", ds=Q,
             deps=[("integer", "rank", "dimension", "ordinal", "none:rank:ok", "None", "ordinal"),
                   ("string", "customer_id", "dimension", "nominal", "none:customer_id:nk", "None", "nominal"),
                   ("real", "if_score", "measure", "quantitative", "avg:if_score:qk", "Avg", "quantitative")],
             rows="[" + Q + "].[none:customer_id:nk]", cols="[" + Q + "].[avg:if_score:qk]",
             mark="Bar", num_format=("[" + Q + "].[avg:if_score:qk]", "*0.000"),
             sort=("[" + Q + "].[none:customer_id:nk]",
                   "[" + Q + "].[avg:if_score:qk]", "DESC"),
             title="50 户全部深夜、%s 户超额" % f["q_all_over"]),
    ]

    twbkit.configure(BI, os.path.join(ROOT, "BankAnomaly_Tableau105.twb"),
                     {F_DS: F_CAP, Q_DS: Q_CAP})
    out = twbkit.build(
        datasources=[
            dict(ds=F_DS, ts=F_TS, cap=F_CAP, csv="bank_fact.csv", fields=fact_fields),
            dict(ds=Q_DS, ts=Q_TS, cap=Q_CAP, csv="bank_queue.csv", fields=q_fields),
        ],
        sheets=sheets,
        dash=dict(name="银行异常交易总览", title="银行异常 · 104.9 万笔 / 17 亿美元",
                  ncols=3, nrows=2),
    )
    twbkit.validate(out)


# ------------------------------------------------------------------
# 3. PowerBI 工程（度量值定义 + 报表页配置）
# ------------------------------------------------------------------
MEASURES = """\
	measure '交易笔数' = SUM('bank_fact'[txn])
		formatString: #,0

	measure '交易金额' = SUM('bank_fact'[amount_m])
		formatString: $#,0.0

	measure '超额占比' = AVERAGE('bank_fact'[over_flag])
		formatString: 0.0%

	measure '深夜占比' = AVERAGE('bank_fact'[late_flag])
		formatString: 0.0%

	measure 'IF 命中率' = AVERAGE('bank_fact'[if_label])
		formatString: 0.00%
"""


def build_pbip(f):
    K.configure(proj="BankAnomaly", logical_id="a1b2c3d4-1111-4a2b-9c3d-000000000003",
                root=ROOT, csv_dir=BI)
    K.build_semantic_model(tables=[
        dict(name="bank_fact", csv="bank_fact.csv", measures=MEASURES,
             desc="交易明细：1,048,567 笔，含时段/城市/金额区间/客户笔数/异常标记",
             money_cols=["amount_m"], pct_cols=[]),
        # bank_queue 不进 Power BI 模型：它是 Top50 排查名单，和 104 万行明细
        # 是不同粒度、建不上关系；没有视觉引用它 → 模型里就是一张没连线的空表。
        # CSV 保留（Tableau 侧在用），要放进报表得先加一个表视觉。
    ])

    F, Q = "bank_fact", "bank_queue"
    vs = [
        K.textbox("v100", 16, 12, 1248, 36, "银行异常交易 · 时段与金额"),
        K.visual("v101", "card", {"Values": [K.meas_ref(F, "交易笔数")]},
                 16, 64, 303, 96, "交易笔数", accent="primary", panel=True),
        K.visual("v102", "card", {"Values": [K.meas_ref(F, "交易金额")]},
                 331, 64, 303, 96, "交易金额（百万）", accent="primary", panel=True),
        K.visual("v103", "card", {"Values": [K.meas_ref(F, "超额占比")]},
                 646, 64, 303, 96, "超额交易占比", accent="alert", panel=True),
        K.visual("v104", "card", {"Values": [K.meas_ref(F, "IF 命中率")]},
                 961, 64, 303, 96, "模型命中率", accent="alert", panel=True),
        K.visual("v105", "clusteredColumnChart",
                 {"Category": [K.col_ref(F, "hour")],
                  "Y": [K.agg_ref(F, "txn", 0)]},
                 16, 172, 764, 532, "20 点最忙（9.3%），4 点最低",
                 accent="primary", panel=True),
        K.visual("v106", "clusteredBarChart",
                 {"Category": [K.col_ref(F, "city_group")],
                  "Y": [K.agg_ref(F, "txn", 0)]},
                 792, 172, 472, 532, "MUMBAI 占 9.9%，9,355 城长尾",
                 accent="primary", panel=True, labels=True,
                 sort=(K.agg_ref(F, "txn", 0), "Descending")),
    ]
    K.build_report(pages=[("ReportSection1", "1 时段与金额", vs)])
    K.validate()


# ------------------------------------------------------------------
# 4. 主流程
# ------------------------------------------------------------------
def main():
    print("[1/4] 数据准备 ...")
    fact, q, f = prep()
    print(f"  {f['rows']:,} 笔 / {f['cust']:,} 客户 / 金额 ${f['amount']:.2f}B "
          f"/ {f['cities']:,} 个城市")
    print(f"  时段峰值 {f['hour_peak']} 点 {f['hour_peak_pct']:.1f}% | 谷值 {f['hour_low']} 点")
    print(f"  {f['city_top']} {f['city_top_pct']:.1f}% | 金额：大额 {f['big_n_pct']:.1f}% 笔数 "
          f"→ {f['big_a_pct']:.1f}% 金额")
    print(f"  超额 {f['over_pct']:.1f}%（${f['over_amt']:.0f}M）/ 深夜 {f['late_pct']:.1f}%（${f['late_amt']:.0f}M）")
    print(f"  单笔客户 {f['single_pct']:.1f}% | IF 命中 {f['if_pct']:.2f}%（金额 {f['if_amt_pct']:.2f}%）")
    print(f"  队列 50 户：深夜 {f['q_all_late']} / 超额 {f['q_all_over']}")

    print("[2/4] 导出 CSV ...")
    p1 = csvkit.export(fact, os.path.join(BI, "bank_fact.csv"))
    p2 = csvkit.export(q, os.path.join(BI, "bank_queue.csv"))
    csvkit.check(p1)
    csvkit.check(p2)

    print("")
    print("[3/4] Tableau 工作簿 ...")
    build_tableau(fact, q, f)

    print("")
    print("[4/4] PowerBI 工程 ...")
    build_pbip(f)
    print("")
    print("完成。")


if __name__ == "__main__":
    main()
