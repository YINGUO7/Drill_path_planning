# 三角形型腔深度强化学习路径规划

本项目使用深度强化学习为圆角三角形型腔规划铣削路径。环境基于
Gymnasium，智能体使用 Stable-Baselines3 的 SAC 算法；策略同时接收低维状态
和型腔覆盖栅格，并通过自定义特征提取网络融合两类信息。

## 项目结构

```text
.
├── src/
│   ├── env.py       # 型腔几何、B-spline 路径、奖励与加工时间评估环境
│   ├── nn.py        # 覆盖栅格 CNN 和多输入特征提取器
│   └── learn.py     # SAC 训练、评估、检查点和 TensorBoard 日志
├── requirements.txt
└── README.md
```

训练生成的 TensorBoard 日志、检查点和模型属于可再生成的实验产物，默认不会
被 Git 跟踪。若某个模型需要长期发布，建议使用 Git LFS、Release 或独立的模型
存储，而不是直接提交到普通 Git 历史中。

## 环境准备

建议使用独立的 Python 虚拟环境。深度学习依赖对最新 Python 版本的支持有时会
滞后，若安装遇到兼容性问题，可优先尝试 Python 3.11。

在 PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 记录的是兼容版本范围，而不是已经验证过的精确环境快照。
完成一次可复现实验后，可额外保存 `python -m pip freeze` 的输出或使用环境锁定
工具记录精确版本。

## 运行

从项目根目录运行环境的随机动作演示：

```powershell
python src/env.py
```

开始 SAC 训练：

```powershell
python src/learn.py
```

训练脚本当前会依次使用多组 `gamma`，每组训练 1,000,000 个时间步。正式运行前，
建议先检查 `src/learn.py` 中的 `gamma_list`、`total_timesteps`、`n_envs` 和评估
频率，并用较小的时间步数做一次冒烟测试。默认使用 CPU；需要 CUDA 时可在当前
PowerShell 会话中设置：

```powershell
$env:SAC_DEVICE = "cuda"
python src/learn.py
```

查看训练指标：

```powershell
tensorboard --logdir sac_milling_nurbs_tensorboard_4
```

## 实验记录建议

每次实验至少记录以下信息：

- Git 提交号；
- 随机种子；
- 环境、奖励函数和 SAC 超参数；
- Python、PyTorch、Gymnasium 与 Stable-Baselines3 版本；
- 覆盖率、加工时间、无效动作数和最终路径图。

代码和实验结果应分开管理：代码、配置和小型汇总表进入 Git；日志、检查点、
权重和大规模中间数据放在 Git 之外。这样既能复现实验，也能保持仓库轻巧。

## 日常 Git 工作流

先检查工作区，再选择性暂存本次相关文件：

```powershell
git status
git diff
git add README.md requirements.txt .gitignore .gitattributes .editorconfig
git diff --cached
git commit -m "完善项目说明和版本控制配置"
```

修改算法后，建议一次提交只表达一个目的：

```powershell
git add src/env.py
git diff --cached
git commit -m "调整完整路径时间奖励下限"
```

提交前始终查看 `git diff --cached`，可以有效避免把无关修改、模型权重或本机
配置带入历史。若暂存了错误文件，可用 `git restore --staged <文件>` 取消暂存；
该命令不会删除工作区中的修改。

