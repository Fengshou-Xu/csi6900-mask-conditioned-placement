# CSI 6900 进度时间线

## 已完成

### 2026-05-13 ~ 05-14 ｜ Proposal 修改

- Objectives walkthrough：删除 Scope 段的 out-of-scope 句子，Objective 3 删 "non-physical"
- Workplan walkthrough：Phase 合并（4→3）、mocap → colored tray、JAX-compatible → JAX-based、语法修正
- 用户自行修改 Workplan 后复查：发现结构问题（两个 Phase 2、周数错位、内部引用错误），逐项修复
- References 修改：按教授要求 "cite conference version"，将 HER/RIG 改为 NeurIPS 格式，Mask 改为 RLC 2026 格式，加 `maxbibnames=99`
- 最终 PDF 检查：发现 3 处遗漏（`produce` → `produced`、残留 "non-physical"、残留 "JAX compatible"），修复后全部通过

### 2026-05-15 ~07:00–08:00 ｜ Enrollment Form 填写

- 讨论 form 各字段内容：Title / Description / Frequency of Meetings / Distribution of Marks
- 用户自写 Description，复查后做了两个小修：复数 + "delivery content" → "deliverables"
- 签名问题：PDF 签名字段要求数字证书，改用 Adobe type 签名确认可行

### 2026-05-15 ~08:00–09:00 ｜ 邮件发送给教授

- 决定在同一 thread 内直接 Reply（而非新建邮件）
- 邮件措辞微调，讨论 `filled in` vs `filled` 语法
- 教授回复时误发了另一位学生 (Tanishk Nandal) 的 form 作为附件
- 教授补发正确附件，下载时遇到 Outlook Safe Attachments 扫描占位符问题，重开 Outlook 后解决

### 2026-05-15 ~09:00–09:30 ｜ 提交注册（Deadline 当天完成）

- 在 uoZone 提交 Service Request：上传签好的 enrollment form + proposal PDF，Status: Submitted
- 回复 grad office (Olivia)，提及当天是 enrolment deadline，礼貌催处理

### 2026-05-15 之前（具体日期不详）｜ 技术准备

- 通读 MuJoCo Playground 源码（`pick.py`、Franka Emika Panda 环境），写了详细的中文注释
- 学习 JAX 基础：JIT、`grad`、`vmap`、autodiff、pytrees
- 构建 JAX benchmark：对比 `jax.lax.scan` 与 Python 循环的速度差异
- 已与教授讨论过 `pick.py` 相关的代码理解

---

## 当前状态（2026-05-19 早上）

- 上一次实质性工作是 05-15 的注册提交
- 05-15 之后未进行代码工作
- 已切换回 Ubuntu 开发环境
- 本地开发环境（MuJoCo Playground）此前已搭建完成，可以运行

---

## 今天要做的事（2026-05-19）

### 下午 3:00 ｜ 与 Professor Bellinger 例行会议（本地时间）

需要在会议前完成以下工作，以展示 Phase 1 的实质性进展：

硬性时间约束：至少预留 1 小时做会议准备；代码工作能完成多少是多少，优先拿到可解释、可展示的进展。

### 会前任务（按优先级排序）

**任务 1：获取环境关键数值**

运行探测脚本，打印出 gripper 在 home 姿态下的 xyz 坐标、finger 关节索引及开合值、actuator 的上下限。这些数值是搭建 placement 环境的前置依赖。

**任务 2：创建 placement-only 环境（`place.py`）**

在 `pick.py` 同级目录下新建文件，通过继承 `PandaPickCube` 并 override 以下方法来实现：

- `reset()`：改为 box 从 gripper 内的稳定夹持状态开始，target 暂时作为抽象 goal point 随机生成在桌面上（而非空中）
- `_get_reward()`：删除 `gripper_box` reward 和 `reached_box` latch，改为纯距离 reward `R_place = 1 - tanh(α · d_t)`；`α` 暂时作为临时超参/config 值处理
- `_get_obs()`：删除 target 旋转误差相关的维度（placement-only baseline 暂不考虑目标朝向）
- `step()`：需要 override，移除或替换 pick-specific 的 latch / success / termination 逻辑，避免依赖 `reached_box` 或 `gripper_box`
- `default_config()`：从 reward scales 中删除 `gripper_box`

**任务 3：测试验证**

运行测试脚本确认：box 初始处于 gripper 可夹住的稳定 grasp 状态、`step()` 不报错、reward 值合理。

**任务 4（锦上添花）：渲染截图**

如果时间允许，render 一帧确认 box 在视觉上确实在 gripper 内，截图作为会议展示材料。

可并行协作：除用户本人外，可让 Opus 4.6 和 DeepSeek V4 Pro 分担代码审查、测试思路或会议材料准备。

---

## Proposal 时间线对照

根据 proposal，项目周期为 2026-05-04 ~ 2026-07-24（12 周）：

| Phase | 周数 | 时间范围 | 内容 | 当前状态 |
|-------|------|----------|------|----------|
| Phase 1 | Week 1–3 | 05-04 ~ 05-24 | 环境搭建 + baseline PPO | **本周结束，剩余 ~5 天** |
| Phase 2 | Week 4–8 | 05-25 ~ 06-28 | Placement mask 设计与训练 | 未开始 |
| Phase 3 | Week 9–12 | 06-29 ~ 07-24 | 评估、对比、报告 | 未开始 |

Phase 1 的 deliverable 是：一个 JAX-based placement-only 环境 + 一个 baseline PPO policy（含 reward、hyperparameters、learning curve 文档）。

目前源码阅读和 JAX 基础学习已完成，属于 Phase 1 的前置工作。环境代码修改是 Phase 1 的核心交付物，今天开始动手。
