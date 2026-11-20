# 剔除复核验证记录

验证对象：`backend/src/services/removal_compute.py`（快速闭式计算）与
`backend/src/services/removal.py` / `backend/src/worker.py`（落库、续跑、公布）。

## 1. 判读与逐次重训一致

`removal_oracle.py` 用 sklearn 的原链条（`StandardScaler().fit` →
`GaussianNB(var_smoothing=...).fit` → `predict`）逐件剔除重训，与快速算法逐条比对
翻转件数。运行：

```bash
python removal_oracle.py
```

覆盖：三类基线、720 条演示规模、5% 离群点、2/4 类、`smoothing` 取 `1e-3/1e-4`、
特征量纲相差 6 个数量级（`centroid` 平移 1e6）、恒定特征列、近单样本类别，
以及 40 组随机形态（重复行、取整造成的近似平票、稀有类、不同折数）。全部一致，
差异件数为 0。

## 2. 集成检查（本地 SQLite + Redis）

- 全部折齐才 `published=true`；中途失败时 `published=false` 且不返回排名。
- 断在第一折后把 `fold_total` 调大模拟中断，再恢复续跑：已落库的折不重算，
  `removal_fold_results` 行数不增加（`(audit_id, fold)` 唯一）。
- 同一复核任务重复投递 5 次（含并发线程）：折行数仍为折数，合计不重复累加。
- 缺类候选：稀有类在两折训练集中各剩 1 条，剔除后该类消失的 2 件样本出现在
  “缺类候选”，不进入排名；其余样本翻转件数与逐次重训完全一致。
- 权限：匿名 GET 401；viewer POST 403；analyst POST 200。
- 已完成复核再次 POST 不重开；失败态再次 POST 续跑。
- 万条样本单折：约 2.2s，峰值 RSS 约 135MB；随机抽 12 件与重训比对一致。

## 3. 口径

翻转件数 = 剔除该条训练样本后，其所在折验证行中判读翻转的条数，跨折累加。
同分按原研究输入顺序（索引升序）。数值平票仅在相对边际 `< 1e-6` 时调用一次真实
重训兜底；该计数在页面的“数值平票重训”列公布。
