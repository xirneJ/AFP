# Stage 3 结果：A0/A1/A2 各 3000 步（真实 ImageNet latent）

> 状态：**已跑完并通过验证；未启动 20k；未跑 FID。**
> 配置见 `docs/STAGE3_FORMAL_TRAINING_PLAN.md` §5。三组唯一差异：
> `model.parameterization`（velocity/state）与 `loss.target_branch`（shared/ema）。

## 1. 运行条件

- 起点：`snapshots/alphaflow-B-2-finetune-init.pt`（released B/2 velocity checkpoint，统一初始化）；
- α schedule：sigmoid，`max_steps=3000`、`change_end_steps=3000`、`gamma=25`、`clamp=0.005`、`ratio_fm=0.5`；
- LR：Adam，lr=1e-4，`num_warmup_steps=75`（修正了 1k 短测中 warmup 未走完的问题），warmup 后恒 LR；
- `ema_halflife_steps=6931`、`ema_rampup_ratio=null`（固定 β=0.9999）；`batch_gpu=16`，global batch 256；
- 硬件：单卡 RTX 5090（**与同事任务共享，0.15–0.18 it/s**），总耗时约 14.3h（A0 4.5h / A1 4.7h / A2 4.8h）。

## 2. 结果

| | A0 velocity+shared | A1 state+shared | A2 state+EMA teacher |
|---|---|---|---|
| steps | 3000/3000 | 3000/3000 | 3000/3000 |
| `loss_total` first→last | 0.998930 → 0.999110 | 0.998940 → 0.999090 | 0.998940 → 0.999080 |
| `loss_total` min / max | 0.499830 / 0.999140 | 0.499900 / 0.999150 | 0.501430 / 0.999150 |
| `grad_norm` first/last/max | 1.2649 / 0.1875 / 1.2649 | 1.4453 / 0.1488 / 1.4453 | 1.4453 / 0.1327 / 1.4453 |
| NaN / Inf | 0 / 0 | 0 / 0 | 0 / 0 |
| 显存峰值 gpumem/reserved | 10.26 / 10.33 GB | 10.26 / 10.33 GB | **11.06 / 11.13 GB** |
| 稳定显存 | ~5.9 / 7.5 GB | ~6.1 / 7.7 GB | ~6.9 / 8.45 GB |
| snapshot | 2.77 GB | 2.77 GB | **3.63 GB** |
| teacher 状态 | — | — | `teacher=True`、`cur_step=3000` |

- **A0 vs A1 loss_total 相对差**：median `2.254e-05`，max `7.163e-04`；
- **A1 vs A2 loss_total 相对差**：median `0.0`，max `6.041e-03`（最大差异出现在 α 过渡段 step≈2001，
  A1=0.50234 vs A2=0.50510，对应 teacher 目标开始生效）；
- `grad_norm < grad_clip_norm=16` 全程未触发裁剪；
- 三个 `snapshot-00003000.pt` 均可加载：A0/A1 `teacher=False`，A2 `teacher=True`，`cur_step=3000`。

### A2 teacher 轨迹（`teacher_student_rel_dist`，单调上升）

| step | 1 | 11 | 101 | 501 | 1001 | 1501 | 2001 | 2501 | 3000 |
|---|---|---|---|---|---|---|---|---|---|
| rel_dist | 0.00002 | 0.00059 | 0.00838 | 0.02022 | 0.02704 | 0.03151 | 0.03213 | 0.03611 | **0.03932** |
| `teacher_ema_updates_total` | 3000（全程） | | | | | | | | |

`teacher_ema_beta=0.9999` 与 `eval_ema_beta` 一致；teacher 每步更新、无梯度。

## 3. 解读

- 三组训练**稳定**：无 NaN/Inf、无 OOM、无梯度爆炸、显存稳定；checkpoint/sampler 正常
  （step 0/1000/2000 均生成 fakes，最终 snapshot 已保存）。
- `loss_total ≈ 1` 是 adaptive loss 归一化的必然结果（每样本权重≈1），**不是**"没学习"；
  中段 `loss_total≈0.5` 与末期 `grad_norm` 下降是 3k α 调度的预期签名。
- A2 的 teacher 距离从 0.00002 单调升到 **0.0393（≈3.9% 相对 L2）**，说明 teacher 目标与学生
  已产生实质差异（对比 1k 短测的 0.0023 —— 后者是 LR warmup 未走完导致的"假冻结"）。
- A1/A2 在 α=1 阶段理论上不调用 teacher（F2 已验证），早期 tick 的 grad 小差异来自
  CUDA 非确定性执行/数据流微小抖动（未开启 deterministic 模式）；两组协议相同，不影响对比。
- **3k 只能证明"实现正确 + 训练稳定 + teacher 生效"**，不能评价生成质量或 EMA 是否提升最终
  FID —— 这些需要 20k（官方比例 α 调度）+ FID 协议。

## 4. 产物

```text
runs/stage3-a0-latent-3k/   log.txt, snapshot-00003000.pt (2.77GB), fakes_*
runs/stage3-a1-latent-3k/   log.txt, snapshot-00003000.pt (2.77GB), fakes_*
runs/stage3-a2-latent-3k/   log.txt, snapshot-00003000.pt (3.63GB, teacher=True), fakes_*
runs/_stage3_3k_report.txt  （脚本自动生成的对比报告）
```

## 5. 下一步（未执行）

- 20k 正式训练：`max_steps=20000`、`change_end_steps=20000`、`num_warmup_steps=500`，
  从 released snapshot 重启；batch 由空闲 profiling 的 samples/sec 决定（默认 b16）；
- FID 只在 20k 结束后按协议执行（val 50k，NFE-1/2，A0/A1/A2 对照）；
- 当前**不启动** 20k / FID。
