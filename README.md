# EEG Arrow Direction Decoding

基于 OpenBCI Cyton+Daisy 平台的脑机接口项目。使用视觉箭头刺激诱发EEG信号，通过机器学习实现4方向单试次解码。

## 项目结构

```
EEG-Arrow-Decoding/
├── openbci/              ← 完整 OpenBCI GUI (Processing/Java, 85 .pde)
│   ├── ArrowStimulus.pde       # 箭头刺激呈现程序
│   ├── StimulusController.pde  # 刺激控制逻辑
│   ├── StimulusLauncher.pde    # 刺激启动器
│   ├── BoardCyton.pde          # Cyton主板通信
│   ├── InterfaceSerial.pde     # 串口通信
│   ├── DataProcessing.pde      # 实时信号处理
│   ├── W_TimeSeries.pde        # 实时波形显示
│   └── ... (共85个.pde)
├── analysis/             ← Python分析脚本
│   ├── plot_region_comprehensive.py  # 15脑区组合解码
│   ├── plot_binary.py                # 二分类分析(6配对)
│   ├── plot_cross_session.py         # 跨Session验证
│   ├── plot_feature_importance.py    # 特征重要性
│   ├── plot_timefreq.py              # 时频特征对比
│   ├── generate_report.py            # Word报告生成
│   └── ... (共28个脚本)
├── reports/              ← 分析报告
│   ├── EEG_Arrow_Stimulus_Technical_Report.md
│   └── binary_classification_report.md
├── README.md
└── .gitignore
```

## 核心链路

```
Processing刺激程序 ──串口标记同步──> OpenBCI采集 ──CSV数据──> Python分析管线
     │                                                          │
     └── 实时EEG波形显示                                  ERP窗口提取
                                                          LDA分类器
                                                          结果可视化
```

## 关键技术

| 层面 | 技术 | 说明 |
|:----|:----|:----|
| 采集 | OpenBCI Cyton+Daisy | 12通道, 500Hz, 24位ADC, SRB2参考 |
| 刺激 | Processing (Java) | 4方向箭头随机呈现, 串口标记同步 |
| 信号处理 | scipy | 1-45Hz带通+50Hz陷波, 零相位滤波 |
| 分类器 | sklearn LDA | 5折交叉验证 + 跨Session泛化验证 |
| 可视化 | Matplotlib | ERP波形, 热力图, 混淆矩阵, 解码准确率 |

## 核心结果

- 4分类准确率: **65.0%** (F+P脑区, 同Session, 机会水平25%)
- 跨Session: **48.2%** (中央区最稳健)
- 左vs右: **95.0%** (额叶)
- 发现轴判别(垂直vs水平)为解码瓶颈(15/15脑区, 最高仅54.5%)

## 使用

**OpenBCI GUI** (刺激呈现):
```
Processing openbci/OpenBCI_GUI.pde
```

**数据分析**:
```bash
cd analysis
python plot_region_comprehensive.py
python plot_binary.py
python generate_report.py
```

## License

MIT
