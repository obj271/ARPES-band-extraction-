# ARPES 能带提取：Gaussian 先验 + MAP

从二维 ARPES 强度图 `I[energy, momentum]` 提取连续能带。以预处理强度作为近似似然，以相邻动量点能量差的 Gaussian 分布作为连续性先验，用动态规划求离散链上的精确 MAP，并用前向–后向算法计算逐点边缘后验。无需训练、标签或 GPU。

支持单带和给定参考色散的多带提取。多带按参考色散的中点划分搜索区域，每条带独立推断；当前支持从低到高排列、互不交叉的参考曲线，不自动判断带数，也不求解交叉处的带身份。三维 `E(kx, ky)` 曲面尚未实现。

## 安装与运行

Python 3.10 或更高版本：

```bash
python -m pip install -e .
python -m arpes_map demo
```

默认生成 **3 条能带**，所有结果直接放在一个 `outputs/` 文件夹中，不再按 `demo`、`smoke` 等命令创建子目录。安装后也可使用 `arpes-map demo`。

```bash
# 单带、双带、三带使用相同输出结构
python -m arpes_map demo --bands 1
python -m arpes_map demo --bands 2
python -m arpes_map demo --bands 3

# 可选缺失列；默认数据完整
python -m arpes_map demo --bands 3 --dropout

# 展宽更大、计数噪声更低的情况
python -m arpes_map demo --bands 3 --gamma-ev 0.027 --peak-counts 1200
```

这些命令会覆盖同一目录中的同名结果；`--output PATH` 可指定另一个运行目录。一次运行的图、曲线、后验和对比指标始终放在该目录下，不创建子目录。

模拟包含 Lorentzian 峰、费米占据、Gaussian 仪器展宽、动量相关强度和 Poisson 计数噪声。默认种子 42，能量/动量采样数 192/100，峰值期望计数 12，本征 Lorentzian 半高半宽 `--gamma-ev 0.009` eV。`--dropout` 将中心四列强度置零；Python 接口使用 `simulate(missing=True)`。

## initial guess 与 truth 的差异

模拟提供独立的近似色散作为 `initial_guess`。它使用不同的能量偏移和曲率，并省略 truth 中的小幅起伏，**不将 truth 直接复制为初始参考**。三带默认模型如下，能量单位为 eV，`k` 单位为 Å⁻¹：

| 能带 | 数据生成的 truth | 近似模型 initial guess |
| --- | --- | --- |
| 1，低能带 | `-0.405 + 0.11 k² + 0.009 sin(3k)` | `-0.385 + 0.085 k²` |
| 2，中间带 | `-0.290 + 0.23 k² + 0.012 sin(5k)` | `-0.275 + 0.200 k²` |
| 3，高能带 | `-0.095 - 0.018 k² + 0.008 cos(4k)` | `-0.080 - 0.025 k²` |

单带使用表中的中间带，双带使用中间带和高能带。`approximate_dispersion(momentum, bands)` 只接收动量和带数，不接收 truth 或强度数据。

当前算法是全局动态规划，没有依赖迭代起点的局部优化。这里的 **initial guess 用于 Gaussian 参考先验及多带搜索分区**，不是数值优化器的初值；`--initial-guess` 与 `--reference` 是等价选项。默认参考先验标准差为 `--reference-sigma 0.04` eV，值越小越依赖参考。

图中同时显示 initial guess、MAP 和 truth，指标分别报告初始参考与各提取方法的误差。truth 只用于评估和绘图，不传入推断，也不用于划分搜索区域。

默认三带、种子 42 的本地结果：

| 方法 | 合并三条带的 RMSE（meV） |
| --- | ---: |
| Initial guess | 14.844 |
| MAP | 2.820 |
| 原始强度逐列峰值 | 6.601 |
| 二维曲率逐列峰值 | 28.441 |
| 最小梯度逐列峰值 | 26.178 |

这是一组固定种子的合成数据结果，不代表实验性能。所有逐列峰值方法使用与 MAP **完全相同的能带搜索分区**，但没有连续性先验，也没有 Gaussian 参考惩罚；曲率和梯度方法的结果是增强图的逐列峰值基线，并非完整的多峰拟合算法。

## 输出

一次默认运行只生成以下六个文件，均位于 `outputs/`：

| 文件 | 内容 |
| --- | --- |
| [map_result.png](outputs/map_result.png) | 原始强度上的多条 MAP、初始参考和 truth；每条带各自的边缘后验及逐点中央 68% 区间 |
| [band_comparison.png](outputs/band_comparison.png) | MAP、原始强度峰值、二维曲率峰值、最小梯度峰值的四栏对比，显示有效列 RMSE |
| [band.csv](outputs/band.csv) | 动量、MAP、后验均值、16%/84% 分位点和有效列标记；多带增加 `band` 列，按能量从低到高编号 1、2、3 |
| [result.npz](outputs/result.npz) | 输入、初始参考、MAP、各方法曲线、增强强度、搜索掩码、似然及边缘后验 |
| [summary.json](outputs/summary.json) | 配置、算法说明、整体及逐带误差；初始参考误差单独记录 |
| [metrics.csv](outputs/metrics.csv) | 每条带、每种方法的全列和有效列 RMSE，单位 meV；仅在有 truth 时生成 |

不再额外生成重复的 `extraction.png` 和 `intensity.png`。没有 truth 的实验输入只生成前五个文件；覆盖运行时会移除上一轮的 `metrics.csv`。

`result.npz` 的主要数组形状（`B` 为带数，`NE` 为能量点数，`NK` 为动量点数）：

| 数组 | 单带 | 多带 |
| --- | --- | --- |
| `intensity`、`curvature_intensity`、`gradient_intensity` | `(NE, NK)` | `(NE, NK)` |
| `band_energy`、`band_indices`、`posterior_mean`、`lower_energy`、`upper_energy` | `(NK,)` | `(B, NK)` |
| `initial_guess`、`reference_energy`、`truth`（如有） | `(NK,)` | `(B, NK)` |
| `raw_band_energy`、`curvature_band_energy`、`gradient_band_energy` | `(NK,)` | `(B, NK)` |
| `likelihood`、`marginals`、`search_masks` | `(NE, NK)` | `(B, NE, NK)` |
| `observed` | `(NK,)` | `(NK,)` |

多带边缘后验沿能量轴归一化：`marginals.sum(axis=1) == 1`；被该带搜索区域排除的状态后验为零。`likelihood` 是施加参考先验和分区之前的局部强度似然，不同带共享相同数据。

缺失列的逐列峰值曲线为 NaN；MAP 依靠先验及两侧证据推断。峰值方法的全列 RMSE 在 JSON 中为 `null`、CSV 中留空，比较时统一使用有效列。整体 RMSE 对所有带、所有选定列的平方误差取平均后开方；逐带指标另列，不做事后带匹配。

## 分析自己的数据

准备线性、有限且非负的强度及严格递增的坐标轴：

```python
import numpy as np

np.savez("data.npz", intensity=intensity_2d, energy=energy_ev, momentum=momentum_inv_angstrom)
```

`intensity.shape == (len(energy), len(momentum))`。坐标轴均至少两个采样点，支持非均匀网格。若原始坐标递减，请连同对应强度维度反转。输入应为线性强度数据，而非截图或对数强度。

没有参考曲线时提取一条主带：

```bash
python -m arpes_map extract --input data.npz
python -m arpes_map extract --input data.npz --energy-window -0.4 -0.1 --anchor 20:-0.25
```

多带需要提供近似色散，按能量从低到高排列，并提前插值到输入动量轴：

```python
# initial_guess_ev.shape == (n_bands, len(momentum_inv_angstrom))
np.savez("guess.npz", momentum=momentum_inv_angstrom, initial_guess=initial_guess_ev)
```

```bash
python -m arpes_map extract --input data.npz --initial-guess guess.npz --reference-sigma 0.04
```

兼容原有 `--reference reference.npz` 和 `band_energy` 数组键。单带参考形状可以是 `(NK,)`。输入 NPZ 自带 `initial_guess` 时自动使用；命令行指定的参考优先。将 `outputs/result.npz` 重新作为输入可复现默认推断；若原运行使用了非默认参数或锚点，需要再次提供相同选项。

`--anchor COLUMN:ENERGY_EV` 是单带软锚点，多带使用 `--anchor BAND:COLUMN:ENERGY_EV`。命令行的 BAND 和 COLUMN 均从 **0** 开始，区别于导出 CSV 的显示编号；锚点不保证轨迹精确经过指定能量。

## 模型与算法

设第 `b` 条带在动量点 `j` 的离散能量为 `z_bj`，参考为 `g_bj`：

```text
P(z_b | I, g) ∝ ∏_j L_j(z_bj)
                × ∏_{j>0} exp[-(z_bj - z_b,j-1)² / (2 σ_j²)]
                × ∏_j exp[-(z_bj - g_bj)² / (2 reference_sigma²)]
                × ∏_j 1[z_bj belongs to band b's search region]
                × optional Gaussian anchor factors
```

无参考的单带省略参考与分区因子。多带分界线是相邻初始参考的逐点中点 `0.5*(g_bj + g_b+1,j)`；分区不重叠，避免各条轨迹重复提取同一强带。参考曲线必须在选定能量范围内且保持严格顺序，每条带每列至少保留一个候选状态。偏差过大的参考可能将真实峰排除在区域外，强重叠或真实带交叉仍可能失败。

预处理依次为：整体缩放、可选 Gaussian 滤波、逐列扣除第 10 百分位背景并将负差截为零、逐列峰值缩放为 1、加入似然底噪，再逐列归一化。底噪公式为 `floor + (1-floor)*contrast`，默认 `floor=1e-3`。原始无能量对比度的列使用均匀似然，并标记 `observed=False`；即使开启动量方向滤波，也保留缺失标记。全图无能量对比度则报错。

均匀动量采样下 `σ_j=eta`；非均匀采样下 `σ_j=eta*sqrt(Δk_j/median(Δk))`。`eta` 是相邻点带能量差的标准差，不是谱峰线宽。能量状态采用均匀离散先验；非均匀能量网格上的边缘概率表示每个候选状态的概率质量。

每条带在给定参考和分区条件下，以动态规划精确求解 MAP，再以前向–后向求和计算归一化边缘后验。没有事后平滑或亚像素插值。多带不是无约束的联合多峰拟合，其后验也未包含参考模型和带数的不确定性。时间复杂度为 `O(B × NK × NE²)`。

联合 MAP、逐点后验均值和逐点区间是不同统计量，逐点区间不必包含联合 MAP。区间使用离散 CDF 的 16%/84% 分位点，不插值。

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--bands`（demo） | 3 | 模拟带数，支持 1、2、3 |
| `--eta` | 0.012 eV | 小值更强地约束相邻能量差，可能压平陡峭色散 |
| `--reference-sigma` | 0.04 eV | 参考先验标准差，小值更依赖 initial guess |
| `--smooth-energy` | 0.7 像素 | 构造似然及增强图前的能量方向滤波 |
| `--smooth-momentum` | 0 像素 | 动量方向滤波 |
| `--background-percentile` | 10 | 背景估计百分位 |
| `--likelihood-floor` | 0.001 | 防止零似然，越大越削弱数据对比度 |
| `--anchor-sigma` | 0.008 eV | 软锚点标准差 |

两种增强图在 MAP 背景扣除之前计算，使用相同的 Gaussian 滤波。导数坐标为 `x=(k-k[0])/median(Δk)`、`y=(E-E[0])/median(ΔE)`，保留非均匀间距；各图独立在全图归一化到 `[0,1]`，缺失列置零。增强图不参与 MAP 推断。

- 二维曲率：固定 `Cx=Cy=1`，计算 `C=((1+fx²)fyy - 2 fx fy fxy + (1+fy²)fxx)/(1+fx²+fy²)^(3/2)`，取 `max(-C,0)`。
- 最小梯度：对八邻域按坐标距离计算方向差商的 RMS 模长 `G`，输出 `f/max(G,ε)`，其中 `ε=max(1e-3*max(G), machine_epsilon)`；边界只使用存在的邻居。

## Python 接口与验证

```python
from arpes_map import extract_band, extract_bands
from arpes_map.synthetic import simulate

# 无参考的单带
single = simulate()
result = extract_band(single["intensity"], single["energy"], single["momentum"])

# 带独立近似参考的多带；返回由 ExtractionResult 组成的 tuple
multi = simulate(bands=3)
results = extract_bands(multi["intensity"], multi["energy"], multi["momentum"],
                        initial_guess=multi["initial_guess"], reference_sigma_ev=0.04)
posterior_band_1 = results[0].marginals
```

```bash
python -m unittest discover -s tests -v
```

测试包括小链穷举验证 MAP 和边缘概率、受限状态的精确后验、多带有偏参考恢复与缺失列、错误参考与锚点验证、统一分区的对比峰值、单/双/三带导出与重新导入，以及改变 truth 不改变推断结果。原有增强算法、非均匀网格、强度单位不变性等检查保留。

近邻 Gaussian 先验与强度似然的思想参考 Xian et al., *A machine learning route between band mapping and band structure*, Nature Computational Science (2022), [DOI](https://doi.org/10.1038/s43588-022-00382-2)。本项目是一维动量链上的简化实现，不声称完整复现论文。增强方法参考 [二维曲率方法](https://arxiv.org/abs/1104.1524)和[最小梯度方法](https://arxiv.org/abs/1612.07880)。

归一化强度是近似似然，不是原始计数的 Poisson 前向模型。后验区间仅描述当前离散模型和参数下的不确定性，未校准实验噪声、仪器分辨率或自能误差。缺失区曲线由先验推断，不能视为观测到新能带。当前仅以合成数据验证，尚未验证真实实验数据。
