# ChromPeak 色谱峰检测工程

这是一个针对一维色谱时间序列的 Python 验证工程。当前目标是先可靠完成预处理、峰位置检测和峰形类型判定，再利用人工真值迭代参数；暂不处理 C++/ARM 移植。

## 当前能力

- 动态噪声：局部一阶差分 MAD、局部 SNR、高噪声工况保护。
- 基线漂移：arPLS、双极性稳健基线及 rolling-ball 局部背景。
- 正峰与负峰：独立检测分支，达到统一置信度阈值的负峰可直接确认。
- 宽峰与鼓包：区分普通正峰、宽峰、鼓包背景上的峰及宽背景候选。
- 重叠峰：根据峰窗口交叠和模板宽度输出重叠/未分离候选。
- 电信号干扰：综合脉冲、宽度、对称性和峰顶平坦度判定。
- 结果解释：逐峰输出分项得分、综合峰置信度和模板匹配置信度。

需求文档中的 5.1～5.6 六类问题、验收指标和数据盘点见 [需求梳理](docs/需求梳理_v1.md)。

## 工程结构

```text
.
├─ configs/                    # 检测阈值、T1～T6组分映射
├─ data/                       # 输入数据工作副本
├─ docs/                       # 需求、开源检索和算法说明
├─ experiments/                # 早期算法比较/试跑脚本，不属于生产主链
├─ outputs/                    # 批量结果、图像和实验输出
├─ src/chrompeak/              # 核心Python包
│  ├─ core.py                  # 数据读取和公共信号工具
│  ├─ detector.py              # 预处理、检测、分类、输出和CLI
│  └─ __main__.py
├─ tests/                      # 关键样例回归测试
├─ pyproject.toml              # 安装和命令行入口
├─ requirements.txt            # 固定依赖版本
└─ run_detector.py             # 无需安装即可运行的入口
```

`.vendor/` 是本地依赖缓存，不作为源码模块使用，也不会提交到版本库。

## 快速运行

工作区已经准备好依赖和输入数据时：

```powershell
python run_detector.py
```

在新环境中：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
chrompeak
```

指定数据、配置和输出目录：

```powershell
python run_detector.py `
  --zip "D:\色谱峰处理算法\一些典型的谱图.zip" `
  --config configs\detector_config.json `
  --component-map configs\component_map.json `
  --out outputs\reliable_detector_results
```

## 统一置信度

`peak_confidence` 表示“该事件是色谱峰”的综合证据，范围 0～1；CSV 同时提供 `peak_confidence_percent`。默认确认阈值是 0.75，正峰和负峰共用同一阈值。

正峰基础权重：

| 分项 | 权重 |
|---|---:|
| 局部 SNR | 25% |
| 相对突出度 | 25% |
| 与参考峰宽的相似度 | 20% |
| 对称性 | 15% |
| 峰顶非平坦度 | 15% |

负峰基础权重：

| 分项 | 权重 |
|---|---:|
| 局部 SNR | 18% |
| 相对突出度 | 15% |
| 峰宽相似度 | 12% |
| 对称性 | 10% |
| 相对基线下探深度 | 20% |
| 原始信号双侧谷深 | 25% |

各项先转换到 0～1，再加权。高噪声、单点脉冲、平顶电干扰、超宽背景和峰间浅谷会施加显式惩罚。所有分项得分均写入结果 CSV，避免只给一个无法解释的最终数字。

`template_confidence` 是另一项独立指标，表示峰与 T1～T6 中某一保留时间模板的匹配程度。它不参与“峰是否存在”的最终确认，避免明显真峰仅因保留时间或低位边界偏移而被错误降级。

阈值位于 [detector_config.json](configs/detector_config.json)：

```json
{
  "confirmation_threshold": 0.75,
  "artifact_threshold": 0.45
}
```

在获得人工真值前，不建议仅凭当前样例继续降低确认阈值。

## 输出文件

默认输出目录为 `outputs/reliable_detector_results/`：

- `all_detected_features.csv`：所有事件及完整分项评分。
- `confirmed_peaks.csv`：综合置信度达到阈值的峰，包括合格负峰。
- `review_required.csv`：证据不足或形态有歧义的事件。
- `interference_candidates.csv`：电尖峰和电干扰候选。
- `confidence_statistics.csv`：按峰类型和状态统计置信度数量、均值、中位数、最小值和最大值。
- `file_quality_summary.csv`：逐文件的峰数、噪声、漂移和置信度摘要。
- `validation_labels_template.csv`：可填写人工真值的完整表。
- `priority_validation_set.csv`：建议优先复核的事件。
- `learned_peak_template.json`：从五条基准谱学习的 T1～T6 模板。
- `plots/`：每条独立曲线的原始信号、基线和带置信度标记的检测图。

## 组分名称

T1～T6 只是按基准谱保留时间排序的模板槽，不代表固定化学组分。请在 [component_map.json](configs/component_map.json) 中填写实际映射：

```json
{
  "T1": "",
  "T2": "",
  "T3": "",
  "T4": "",
  "T5": "",
  "T6": ""
}
```

## 测试

```powershell
python -m unittest discover -s tests -v
```

回归测试覆盖：六个参考模板槽、F1 强负峰/强正峰确认、H3 平顶电干扰排除，以及 B10 高噪声下的强峰保留。

## 当前验证状态

- 输入 45 个 CSV，去除 4 组完全重复文件后为 41 条独立曲线。
- 五条基准谱均学习到 6 个稳定模板峰。
- F1：0.3267 min 负峰置信度约 97.8%，已确认；0.3983 min 强正峰约 94.3%，已确认并标记为重叠正峰。
- F1：2.355 min 峰约 98.1%，已确认。
- H3：4.3633 min 平顶事件约 36.0%，保持电信号干扰/伪峰。

这些百分比是当前规则体系下的“概率式综合分数”，不是经过大规模标注集校准后的真实统计概率。要证明文档中的检出率、误报率和定量误差指标，仍需用人工标注模板建立逐峰真值并进行盲测。

## 下一步

1. 填写 T1～T6 组分映射和允许负峰的组分/通道信息。
2. 优先核验 `priority_validation_set.csv`，填写人工真值模板。
3. 根据真值校准确认阈值和分项权重，输出 ROC/PR、漏检率、误报率和类型混淆矩阵。
4. 真值稳定后再完善重叠峰解卷积、面积定量和 C++/ARM 工程化。
