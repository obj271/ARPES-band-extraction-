# ARPES 能带提取：Gaussian 先验 + MAP

从二维 ARPES 强度图 `I[energy, momentum]` 提取一条连续能带。无需训练、标签或 GPU；以预处理强度作为近似似然，以相邻动量点能量差的 Gaussian 分布作为连续性先验，使用动态规划求**整个离散能量网格上的全局 MAP 路径**。

当前版本处理 `E(k)` 切片中的一条目标带，提供模拟验证、可选参考色散和软锚点、逐点边缘后验、CSV/NPZ 导出及 PNG 图。多带图可通过能量窗口或参考色散逐条分析；这不保证交叉处的带身份，也不自动判定带数。三维 `E(kx, ky)` 曲面尚未实现。

## 安装和快速演示

Python 3.10 或更高版本：

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -e .
python -m arpes_map demo --output outputs/demo
```

也可 `python -m pip install -r requirements.txt` 后在仓库根目录运行模块。安装项目后可使用等价命令 `arpes-map demo`。

模拟包含 Lorentzian 峰、费米占据、仪器展宽、动量相关强度和 Poisson 计数噪声。默认生成完整数据，不在动量 0.0 附近人为制造缺失。添加 `--dropout` 可将中心四列强度置零，用于验证探测器缺失时的恢复：

```bash
python -m arpes_map demo --dropout --output outputs/dropout
```

Python 接口同样默认生成完整数据，使用 `simulate(missing=True)` 启用缺失；原有 `--no-dropout` 选项仍可使用。真值仅用于评估和绘图，不传入提取器。默认种子 42，能量/动量采样数 192/100，峰值期望计数 12。`summary.json` 报告 MAP RMSE、有效列 RMSE 和相同有效列上的原始逐列 argmax 基线，单位 meV。

生成展宽更大、噪声更小的另一版强度图：

```bash
python -m arpes_map demo --gamma-ev 0.027 --peak-counts 1200 --output outputs/broadened_low_noise
```

`--gamma-ev` 为本征 Lorentzian 半高半宽（HWHM），默认 0.009 eV；上例增大到 0.027 eV（本征 FWHM 为 54 meV），仪器 Gaussian 展宽保持原值。峰值期望计数从 12 提高到 1200，使相同相对强度处的 Poisson 相对噪声约降至原来的十分之一。Python 接口可使用 `simulate(gamma_ev=0.027, peak_counts=1200)`；模拟参数保存于 `summary.json`。

已保存的示例：[模拟强度图](outputs/broadened_low_noise/intensity.png)、[三种方法提取对比](outputs/broadened_low_noise/band_comparison.png)、[误差与参数](outputs/broadened_low_noise/summary.json)及[完整数据](outputs/broadened_low_noise/result.npz)。

## 分析自己的数据

准备 `.npz` 文件：

```python
import numpy as np

np.savez("data.npz", intensity=intensity_2d, energy=energy_ev, momentum=momentum_inv_angstrom)
```

`intensity.shape == (len(energy), len(momentum))`，数值须有限且非负，使用线性强度。两条坐标轴须有限、严格递增且至少有两个采样点；能量单位 eV，动量单位 Å⁻¹。允许非均匀网格。若原始坐标递减，请连同对应的强度维度一起反转。不要将颜色截图、对数强度或含负数的背景扣除结果直接当作输入。

```bash
python -m arpes_map extract --input data.npz --output outputs/real --eta 0.012
```

限制能量窗口，避免追踪到其他强带：

```bash
python -m arpes_map extract --input data.npz --energy-window -0.4 -0.1 --anchor 20:-0.25 --output outputs/selected
```

`--anchor COLUMN:ENERGY_EV` 可重复指定；COLUMN 从 0 开始，是输入动量列索引。锚点是 Gaussian 软约束，默认标准差 `--anchor-sigma 0.008` eV，不保证精确经过指定值。

如有 DFT 近似色散或手工给出的目标曲线：

```python
np.savez("reference.npz", momentum=momentum_inv_angstrom, band_energy=reference_energy_ev)
```

```bash
python -m arpes_map extract --input data.npz --reference reference.npz --reference-sigma 0.04 --output outputs/guided
```

参考曲线须预先插值到相同动量坐标，`band_energy` 形状为 `(n_momentum,)`。它作为显式 Gaussian 能量先验参与目标函数，并非优化初值；标准差越小越依赖参考，可能将参考误差带入结果。

## 模型与算法

设每个动量点的带能量为离散变量 `z_j ∈ energy`：

```text
P(z | I) ∝ ∏_j L_j(z_j)
           × ∏_{j>0} exp[-(z_j - z_{j-1})² / (2 σ_j²)]
           × optional reference and anchor Gaussian factors

z_MAP = argmin_z { -Σ_j log L_j(z_j)
                  + Σ_{j>0} (z_j - z_{j-1})² / (2 σ_j²)
                  + optional Gaussian penalties }
```

预处理依次为：整体缩放、可选 Gaussian 滤波、逐列扣除第 10 百分位背景并将负差截为零、逐列缩放峰值为 1、加入正的似然底噪，最后逐列归一化。底噪使用 `floor + (1-floor) × contrast`，默认为 `1e-3`。无能量对比度的原始列使用均匀似然，并在输出中标记 `observed=False`。即使开启动量方向滤波，也保留这些缺失标记；全图无能量对比度则报错。

均匀动量采样下 `σ_j = eta`。非均匀采样下采用随机游走约定：`σ_j = eta × sqrt(Δk_j / median(Δk))`。`eta` 是**相邻点带能量差**的标准差，不是峰的 Gaussian 线宽。改变采样密度后应重新选择 eta。能量轴上的状态采用均匀离散先验；对于非均匀能量网格，结果依赖所用候选采样点，边缘概率表示每个候选状态的质量。

动态规划递推保存每个能量状态的最佳前驱，再反向恢复完整路径；精确求解当前一维链上的离散 MAP，无局部极值初始化问题。不会进行改变目标函数的事后平滑或亚像素插值。时间复杂度 `O(n_k × n_E²)`，内存 `O(n_k × n_E + n_E²)`；大数据建议先选择合理的能量窗口。

前向–后向算法另外计算归一化边缘后验。联合 MAP、逐点后验均值和逐点区间是不同统计量，逐点区间不必包含联合 MAP。区间使用离散 CDF 的 16% 和 84% 分位点，不插值。

### 主要参数

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--eta` | 0.012 eV | 小值更强地约束相邻能量差；可能压平陡峭色散 |
| `--smooth-energy` | 0.7 像素 | 构造似然前的能量方向滤波；0 为关闭 |
| `--smooth-momentum` | 0 像素 | 构造似然前的动量方向滤波 |
| `--background-percentile` | 10 | 逐列背景估计百分位 |
| `--likelihood-floor` | 0.001 | 防止零概率；越大越削弱数据对比度 |
| `--reference-sigma` | 0.04 eV | 目标色散 Gaussian 先验标准差 |
| `--anchor-sigma` | 0.008 eV | 所有软锚点的 Gaussian 标准差 |

用 `python -m arpes_map extract --help` 查看完整参数。建议比较多个 eta（如 0.006、0.012、0.024 eV）下的轨迹，并独立检查选定位置的 EDC/MDC 峰。

## 输出

| 文件 | 内容 |
| --- | --- |
| `extraction.png` | 四幅对比图：原始强度叠加 MAP、边缘后验及逐点中央 68% 区间、二维曲率强度、最小梯度强度 |
| `intensity.png` | 独立的原始强度图，不叠加提取曲线 |
| `band_comparison.png` | 三种提取结果并排对比：MAP、曲率增强图逐列峰值、最小梯度增强图逐列峰值；模拟时显示真值和有效列 RMSE |
| `band.csv` | 动量、MAP 能量、后验均值、区间上下界、有效列标记 |
| `result.npz` | 输入坐标及强度、`curvature_intensity`、`gradient_intensity`、似然、MAP 索引/能量、边缘后验、均值/区间、有效列标记 |
| `summary.json` | 推断参数、输入来源或模拟种子、形状、目标函数值、强度对比方法及归一化设置；模拟时另含误差指标 |

同一输出目录中同名文件会覆盖。输出能量保持 eV；模拟 RMSE 使用 meV。默认演示的 `result.npz` 另含 `truth` 和 `clean`，可复查。可选参考曲线保存为 `reference_energy`。

`result.npz` 还保存 `curvature_band_energy` 和 `gradient_band_energy`。这两条曲线分别取对应增强图各动量列的全局峰值，无能量对比度的列记为 NaN；未使用 MAP 连续性先验。`summary.json` 记录其有效列 RMSE。逐列峰值可能选中噪声或背景，尤其是最小梯度图；三栏图保留这些结果以便直接比较。

两种对比强度均为与原始 `intensity` 同形状的二维数组，`demo` 和 `extract` 默认生成。它们使用全图峰值缩放后的输入和相同的 `--smooth-energy` / `--smooth-momentum` Gaussian 滤波，在 MAP 背景扣除之前计算。导数坐标为 `x=(k-k[0])/median(Δk)`、`y=(E-E[0])/median(ΔE)`，以像素等效单位平衡两个方向，并保留非均匀采样间距。

- **曲率强度**：采用 [Zhang et al. 的二维曲率公式](https://arxiv.org/abs/1104.1524)，固定 `Cx=Cy=1`。对滤波强度 `f` 计算 `C=((1+fx²)fyy - 2 fx fy fxy + (1+fy²)fxx)/(1+fx²+fy²)^(3/2)`，取 `max(-C, 0)`，使峰脊显示为亮色。
- **梯度强度**：采用 [He et al. 的最小梯度法](https://arxiv.org/abs/1612.07880)。在每个像素的八邻域，按坐标距离计算方向差商并取 RMS 模长 `G`，输出 `f/max(G, ε)`，其中 `ε=max(1e-3 × max(G), machine_epsilon)`；边界仅使用存在的邻居。均匀网格内部的 RMS 与论文八方向模长相差固定倍数，在最终归一化时消除。

两幅增强图分别在全图归一化到 `[0, 1]`，全零结果保留为零；无能量对比度的原始列置零。色条表示各自的相对增强强度，不能用于比较绝对光电子计数。图中叠加同一条 MAP 曲线，模拟时另叠加真值，便于检查增强峰脊的位置；增强图仅作为输出对比，不参与 MAP 推断。

## Python 接口

```python
from arpes_map import comparison_intensities, extract_band

result = extract_band(intensity_2d, energy_ev, momentum_inv_angstrom, eta_ev=0.012)
band = result.band_energy
posterior = result.marginals
comparison = comparison_intensities(intensity_2d, energy_ev, momentum_inv_angstrom)
curvature = comparison["curvature_intensity"]
gradient = comparison["gradient_intensity"]
```

## 验证

```bash
python -m unittest discover -s tests -v
```

测试以小链穷举的 MAP 和精确边缘概率验证两套推断算法，并检查先验强弱极限、模拟恢复、缺失列、强度单位不变性、目标带约束、非法输入和 CLI 导出/重新导入。强度对比另验证解析曲面的曲率、非均匀网格八邻域梯度、峰位置、缺失列及滤波参数。GitHub Actions 在 Python 3.10/3.12 上配置相同测试及演示。

本次本地验证使用 Python 3.14.2、NumPy 2.5.4、SciPy 1.18.1、Matplotlib 3.11.2，16 项测试全部通过。默认完整模拟的 MAP 全列及有效列 RMSE 均为 **2.569 meV**，原始逐列 argmax 为 **34.101 meV**。这是一组固定种子的合成数据结果，不是实验性能或跨数据集基准；不表示达到相应实验分辨率。

## 解释范围与文献

近邻 Gaussian 先验与强度似然的思想参考 Xian et al., *A machine learning route between band mapping and band structure*, Nature Computational Science (2022), [DOI: 10.1038/s43588-022-00382-2](https://doi.org/10.1038/s43588-022-00382-2)，尤其是式 (2)、(4)。该论文处理二维动量网格上的三维色散，并使用局部迭代及物理初始化；本项目是针对一维动量切片的独立简化实现，使用链结构的精确动态规划，不声称完整复现论文。

归一化强度是约定的**近似似然**，不是原始计数的 Poisson 前向模型。边缘后验区间仅描述当前离散模型和超参数下的不确定性，未校准实验噪声、仪器分辨率或自能误差。缺失区的曲线由先验推断，不能视为观测到新能带。强重叠、带交叉、费米截断或矩阵元抑制可能导致跳带或偏差；MAP 轨迹也不等同于裸能带或材料完整电子结构。当前仅用合成数据验证，尚未验证真实实验数据。
