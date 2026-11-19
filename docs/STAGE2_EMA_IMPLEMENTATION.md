# Stage 2 实现说明：EMA Target Teacher（A2）

> 状态：**代码已实现，F 组测试待运行**。本轮未启动任何 EMA 训练。
> 设计依据：[docs/STAGE2_EMA_TEACHER_DESIGN.md](STAGE2_EMA_TEACHER_DESIGN.md)。

---

## 1. 修改文件清单

| 文件 | 改动 |
| --- | --- |
| `configs/loss/alphaflow.yaml` | 新增 `target_branch: shared`（默认）。`shared` = 官方行为（学生 + stop-grad）；`ema` = A2（独立 EMA teacher 构造目标）。 |
| `src/training/loss.py` | `AlphaFlowLoss.forward` 增加目标网络选择：`target_branch=ema` 时所有**目标构造**（`_compute_velocity_cfg`、`_compute_mean_velocity_c`、`_compute_mean_velocity_d`，含 α=0 的 JVP、CFG 的 cond/uncond、clamp）使用 `self.teacher`；主预测仍用 student。数学公式、权重、分支逻辑均未改。 |
| `src/training/network_utils.py` | ① `init_ema_teacher_from_student()`：在 `load_states` **之后** `deepcopy(student)`，`eval().requires_grad_(False)`；② `init_modules` 中按 `loss.target_branch=ema` 创建 teacher，并从 snapshot 的 `teacher` 字段恢复（缺失则 warn 并从当前 student 初始化）；③ `save_snapshot(..., loss_fn=None)` 新增 `teacher` 字段；④ `teacher_student_distance_()` 诊断工具。 |
| `src/train.py` | ① 新增 teacher 参数缓存；② 在**与 eval EMA 相同的位置/频率**（optimizer.step + grad clip 之后、`stats.cur_step % ema_update_freq == 0`）用同一个 `update_ema` 更新 teacher；③ tick 日志新增 `teacher_student_rel_dist`（A2 才出现）；④ `save_snapshot`/`evaluate_and_snapshot` 传入 `loss_fn` 以便保存 teacher。 |
| `scripts/check_state_equivalence.py` | 新增 **F 组**（F1–F10）验证 EMA teacher 行为（见 §4）。main 中追加 `section_f_ema_teacher`。 |

**未改动**：`src/inference.py`、`scripts/generate.py`、`scripts/evaluate.py`、采样器、`alphaflow` 损失公式/权重、α/ratio_fm 调度、CFG/clamp 语义、数据管线。

---

## 2. 使用方式（A0 / A1 / A2）

```bash
# A0  velocity + shared stop-grad（默认，不传参数）
# A1  state + shared stop-grad
#   model.parameterization=state
# A2  state + EMA teacher
#   model.parameterization=state loss.target_branch=ema
```

A2 除上述两行外，其余 override（起点、步数、lr、batch、调度、评估 EMA、NFE 协议）与 A1 完全一致。
例（正式实验模板，本轮不执行）：

```bash
bash scripts/train_single_gpu.sh alphaflow-sigmoid-latentspace-B-2-single-gpu \
  desc=stage2-a2-state-ema \
  model.parameterization=state \
  loss.target_branch=ema \
  training.resume.on_start_ckpt_path="$PWD/snapshots/alphaflow-B-2-finetune-init.pt" \
  training.resume.whole_state=false training.resume.allow_missing_keys=true \
  model.ema_rampup_ratio=null \
  training.max_steps=<budget> training.steps_per_tick=10 \
  ...
```

- teacher 只用于 loss 目标；生成/FID 仍统一使用 `nets.ema`（A0/A1/A2 评估口径一致）。
- A2 的内存/快照额外开销：约 +1 份模型参数（B/2 约 0.5GB 显存 + 快照多 ~0.5GB）。
- tick 日志中 A2 会多出 `teacher_student_rel_dist=<rel L2>`（诊断 teacher 是否在移动）。

---

## 3. 实现的精确语义（与设计文档一致）

1. **更新时序**：`Forward → Backward → Gradient Clip → Optimizer Step → EMA Update`；教师与 eval EMA
   在同一位置、同一 `ema_update_freq` 条件、同一 `stats.cur_step`（= optimizer step 计数，**不是扩散时间**）
   更新；decay 公式 `β=0.5^(1/H_t)`、`H_t=min(H, step·rampup_ratio)` 完全复用 `update_ema`。
2. **初始化**：teacher 在 `load_states` 之后 deepcopy → step 0 时 `teacher == student` 逐位一致。
3. **两个头/共享 trunk**：teacher 的**所有**浮点参数与 bufffers 统一 EMA；目标只使用 teacher 的
   velocity 头（与 A1 的 velocity 空间目标一致），state 头仅 EMA/保存。
4. **不参与优化**：teacher 不在 optimizer、不参与 grad clip/DDP；backward 后 `.grad is None`。
5. **分支**：
   - d 分支：`u_tgt = α·v_cfg + (1−α)·u_θ̄(z_s,r,s)`，再 `clip(±clamp_utgt)`，`f_tgt=z_t−(t−r)u_tgt`；
   - α=0：`u_tgt = v_cfg − (t−r)·D_t u_θ̄`（**基础速度与 JVP 导数都来自 teacher**，helper 内 JVP 对
     传入的 `target_net` 求导，因此不存在 teacher 基础速度 + student 导数的混合）；
   - α=1：跳过目标分支（`u_tgt=clip(v_cfg)`），**不调用 teacher**；
   - FM：velocity 头主预测 + `u_tgt=clip(v_cfg)`，**不调用 teacher**（非 cfg 配方）；
   - CFG（κ≠0）：`velocity_cfg` 的 cond/uncond 网络调用使用 teacher（目标构造的一部分）；
   - clamp/adaptive weighting：不变。
6. **快照**：新增 `teacher` 字段；resume 时从快照严格恢复；从无 `teacher` 的检查点继续训练会 warn 并
   用当前 student 初始化（等于重新起跑 teacher 轨迹）。

---

## 4. F 组验证清单（`scripts/check_state_equivalence.py → section F`）

| # | 检查 | 判据 |
| --- | --- | --- |
| F1 | teacher 与 student 初始化逐位一致（params + buffers），teacher frozen/eval | `torch.equal` 全部通过；`requires_grad=False`、`training=False` |
| F2 | A1 vs A2 初始化时**各分支** target、per-sample loss 一致；teacher 的调用分支正确 | target/loss max rel diff < 1e-5；fm/α=1 分支 teacher 调用数 = 0；d/α=0/mixed 分支 > 0 |
| F3 | student 梯度 A1/A2 一致；teacher 无梯度 | 全部梯度张量一致（<1e-5）；teacher `.grad` 全 None |
| F4 | teacher 不进入 optimizer | 参数 id 不与 optimizer 交集；`init_optimizers` 源码只用 `nets.net` |
| F5 | EMA 公式/rampup/时序 | β==0.5^(1/H)；teacher==β·old+(1−β)student；rampup `H_t=min(H,step·ratio)`；源码顺序 `optimizer.step()` 先于 teacher 更新且用 `cur_step % ema_update_freq` |
| F6 | α=0 JVP 的基础速度与导数都来自 teacher（非混合） | A2 的 JVP 导数 ≠ student JVP（扰动 teacher 后 >1e-3）；"teacher 基础 + student JVP"混合量与 A2 不同 |
| F7 | CFG cond/uncond 目标路径均用 teacher | teacher-guided velocity ≠ student-guided（>1e-3）；teacher forward 次数 ≥ 3（uncond+cond+u⁻） |
| F8 | α=1 与 FM 不错误调用 teacher | teacher forward 调用数 = 0 |
| F9 | checkpoint 保存/加载/往返正确 | 保存→加载 teacher state 逐位一致；save/load 源码包含 teacher 字段 |
| F10 | evaluation EMA 与 target EMA 独立、不混用 | 两模块 object identity 不同；更新 teacher 不改 eval EMA；扰动 eval EMA 不改 A2 loss；快照中 net/ema/teacher 为不同状态 |

此外原有 A–E 组（Stage 1 的 173 项等价性检查）继续运行，因此一次完整命令同时回归 Stage 1。

运行（服务器，仓库根目录）：

```bash
cd ~/qiangchen/AFP
git pull
python scripts/check_state_equivalence.py
```

期望末尾：`RESULT: ALL <N> CHECKS PASSED`（FAIL 会打印数值差异与名称）。

### 实测结果（服务器 RTX 5090，commit `c949bd0`）

```
RESULT: ALL 216 CHECKS PASSED      （173 项 Stage-1 A–E 回归 + 43 项 F）
```

| 检查 | 实测 |
| --- | --- |
| F1 初始化逐位一致 / frozen | params_equal=True, buffers_equal=True；eval + requires_grad=False |
| F2 A1/A2 target & loss（7 种模式） | 全部 **0.000e+00**；teacher 调用数：d=1、α=0=1、mixed=2、d_all=1、α0_all=1、fm=0、α=1=0 |
| F3 student 梯度一致 / teacher 无梯度 | 35 tensors，max abs diff **5.588e-09**；teacher grads = none |
| F4 optimizer 隔离 | teacher 参数不在 optimizer；`init_optimizers` 只用 `nets.net` |
| F5 decay / lerp / rampup / 时序 | β=0.999899998193 与公式一致；lerp max abs err **7.451e-09**；rampup step=10 → β=0.25；源码顺序 `optimizer.step()` 在 teacher 更新之前，且使用 `cur_step % ema_update_freq` |
| F6 α=0 JVP 非混合 | teacher-JVP 与 student-JVP 显著不同（rel diff 1.573e+03）；"teacher 基础 + student JVP"混合量与 A2 不重合（0.782e-04 < 1e-3 容限） |
| F7 CFG cond+uncond 均走 teacher | teacher-guided velocity 与 student-guided 显著不同（3.383e+03）；teacher forward 次数 = **3**（uncond + cond + u⁻） |
| F8 α=1 / FM 不误调 teacher | forward 次数 = **0** / **0** |
| F9 checkpoint 往返 | teacher state 保存→加载逐位一致；save/load 源码含 teacher 字段 |
| F10 eval EMA 与 target EMA 分离 | 两模块独立；teacher 更新不改 eval EMA；扰动 eval EMA 对 A2 loss 影响 **0.000e+00**；快照三状态互异（max \|ema−teacher\|=5.268e-02） |

同时 A–E 组 173 项的数值与 Stage 1 验收时逐项一致（例如 A 组 realizable target rel diff 0.0、
C0 全网络映射 0.0、D 组逐步骤采样 0.0、E 组分支组合 ≤2.4e-07），说明默认 `target_branch=shared`
完全保持了 A0/A1 的既有行为。

结论：EMA teacher 的实现（含 teacher 创建/更新/保存/恢复、目标分支切换、JVP/CFG/α=1/FM 分支语义、
评估 EMA 与 target EMA 分离）已通过全部静态与数值验证。**尚未进行任何 A2 训练。**

---

## 5. 当前未做 / 已知限制

- **未启动任何 EMA 训练**（A2 未跑过 1 步真实训练；只有 F 组静态/数值验证）。
- 未添加 A2 的 experiment registry preset（本轮先用 `loss.target_branch=ema` override；正式实验前建议加 preset）。
- 多卡 teacher 参数 broadcast 未实现（当前限定单卡；多卡需额外同步兜底）。
- FSDP 下的 `teacher_student_rel_dist` 需要 `summon_full_params`，当前未支持（单卡 null 策略无影响）。
- 真实 ImageNet latent sanity 仍按 [LATENT_VALIDATION_PLAN.md](LATENT_VALIDATION_PLAN.md) **单独**进行，不与 EMA 实现混合执行。
- 任何训练（包括 A2 的 1–2k sanity）都要等 F 组全部通过并经审核批准后再启动。
