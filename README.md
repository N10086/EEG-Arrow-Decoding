# EEG Arrow Direction Decoding

基于 **OpenBCI Cyton+Daisy** 平台的完整脑机接口项目。使用 Processing 编写视觉箭头刺激程序，通过 OpenBCI 采集 12 通道 EEG 信号，基于 Python + scikit-learn 实现 4 方向（⬆⬇⬅➡）单试次解码，并系统性对比了 9 种特征工程方案与 15 种脑区组合。

---

## 目录

- [项目结构](#项目结构)
- [工作流程](#工作流程)
- [核心结果](#核心结果)
- [openbci/ — 完整 OpenBCI GUI 源代码](#openbci--完整-openbci-gui-源代码)
- [analysis/ — Python 分析脚本详解](#analysis--python-分析脚本详解)
- [reports/ — 分析报告](#reports--分析报告)
- [环境依赖](#环境依赖)
- [运行方法](#运行方法)

---

## 项目结构

```
EEG-Arrow-Decoding/
├── openbci/              ← 完整 OpenBCI GUI 源代码 (85 个 .pde 文件)
├── analysis/             ← Python 数据分析脚本 (28 个)
├── reports/              ← 分析报告 (Markdown)
├── README.md
└── .gitignore
```

---

## 工作流程

```
┌──────────────────────┐
│  Processing 刺激程序  │  ArrowStimulus.pde, StimulusController.pde
│  4方向箭头随机呈现    │  200ms 刺激 + 800-1000ms 间隔
│  串口标记同步         │  标记值 2.0001-2.0004 写入第33列
└─────────┬────────────┘
          │ 串口通信
┌─────────▼────────────┐
│  OpenBCI Cyton+Daisy  │  12 通道, 500Hz, 24位 ADC, SRB2 参考
│  EEG 数据采集         │  每个 Session 200 trials (50/方向)
│  5 个 Session         │  总计 ~1000 trials
└─────────┬────────────┘
          │ CSV 数据文件
┌─────────▼────────────┐
│  Python 信号处理       │  带通滤波 (1-45Hz) + 50Hz 陷波
│  ERP 窗口特征提取      │  P1(80-130ms) N1(140-200ms) P2(200-300ms) P3(300-500ms)
│  LDA 分类器            │  4 分类 + 5折交叉验证
│  可视化 + 报告生成     │  Matplotlib + python-docx
└──────────────────────┘
```

---

## 核心结果

| 指标 | 最佳结果 | 脑区 | 说明 |
|:----|:--------|:----|:----|
| 4 分类准确率（同 Session） | **65.0%** | F+P (额+顶) | 机会水平 25% |
| 4 分类准确率（跨 Session） | **48.2%** | Central (中央) | 训练 S1-4, 测试 S5 |
| 左 vs 右 | **95.0%** | 额叶 / F+P | 几乎完美区分 |
| 上 vs 下 | **72.0%** | F+P+O | 垂直方向最难区分 |
| 轴判别（垂直 vs 水平） | **54.5%** | F+P | ⚠ 解码瓶颈 |

**关键发现**: 轴判别（垂直 vs 水平）在 **15/15个脑区组合** 中均为最弱环节，是 4 分类无法突破 65% 的根本原因。

---

## openbci/ — 完整 OpenBCI GUI 源代码

`openbci/` 包含完整的 OpenBCI GUI 项目（85 个 Processing .pde 文件），基于 [OpenBCI_Processing](https://github.com/OpenBCI/OpenBCI_GUI) 二次开发，新增了箭头视觉刺激模块。

### 箭头刺激模块（本项目核心开发）

| 文件 | 功能 |
|:----|:----|
| `ArrowStimulus.pde` | **箭头刺激主程序**。在屏幕中央绘制 4 方向箭头（Up/Down/Left/Right），随机呈现 200ms，通过串口向 OpenBCI 发送标记值（2.0001-2.0004）实现 EEG 数据与刺激的精确同步 |
| `StimulusController.pde` | **刺激控制器**（~1900 行）。管理刺激序列生成（伪随机排列、方向均衡）、试次流程控制（刺激呈现→间隔→循环）、与 OpenBCI 主界面的联动逻辑 |
| `StimulusLauncher.pde` | **刺激启动器**。提供实验参数配置界面（试次数量、刺激时长、间隔时长），一键启动实验 |

### OpenBCI 核心模块

| 文件 | 功能 |
|:----|:----|
| `OpenBCI_GUI.pde` | **程序主入口**。Processing 主程序入口，管理窗口创建、全局初始化、各模块加载 |
| `BoardCyton.pde` | **Cyton 主板驱动**（~550 行）。与 OpenBCI Cyton 主板的完整通信协议实现，包括数据解析、通道配置、增益设置、阻抗检测等 |
| `InterfaceSerial.pde` | **串口通信**（~560 行）。管理串口连接/断开、波特率设置、数据流控制。支持 WiFi Shield 和本地串口两种连接模式 |
| `DataProcessing.pde` | **实时信号处理**（~400 行）。在线滤波、FFT 计算、包络检测。支持 Butterworth 带通/带阻/高通/低通滤波器 |
| `DataSource.pde` | **数据源抽象层**。定义统一的 EEG 数据获取接口，支持实时采集和离线回放两种模式 |

### 数据记录模块

| 文件 | 功能 |
|:----|:----|
| `DataLogger.pde` | **数据记录器**。将实时 EEG 数据写入 CSV 文件，包含时间戳、16通道原始值、加速度计数据、标记通道 |
| `DataWriterODF.pde` | **OpenBCI 数据格式写入器**（~100 行）。生成符合 OpenBCI 标准的 CSV 数据文件（含 33 列，第 33 列为刺激标记） |
| `DataWriterBDF.pde` | **BDF 数据格式写入器**（~1250 行）。支持将数据保存为 EDF/BDF 标准生物信号格式，兼容 EEGLAB 等专业工具 |
| `DataWriterBF.pde` | **BrainFlow 数据格式写入器**。兼容 BrainFlow 生态 |
| `DataWriterAuxODF.pde` | **辅助数据写入器**。额外记录模拟引脚和数字引脚数据 |
| `DataSourcePlayback.pde` | **数据回放源**。加载已有数据文件进行离线回放 |
| `DataSourceSDCard.pde` | **SD 卡数据源**。从 OpenBCI 板载 SD 卡读取数据 |

### 实时显示 Widget（可视化组件）

| 文件 | 功能 |
|:----|:----|
| `W_TimeSeries.pde` | **实时波形显示**（~1300 行）。多通道 EEG 波形实时滚动绘制，支持缩放、平移、通道选择、颜色配置 |
| `W_FFT.pde` | **频谱分析**（~300 行）。实时 FFT 频谱绘制，显示各频段功率分布 |
| `W_Spectrogram.pde` | **语谱图**（~550 行）。时频域的 2D 热力图显示，x=时间, y=频率, 颜色=功率 |
| `W_HeadPlot.pde` | **头部拓扑图**（~1550 行）。实时绘制头皮电位分布拓扑图，支持插值算法 |
| `W_BandPower.pde` | **频带功率显示**。Delta/Theta/Alpha/Beta/Gamma 五个频带的功率柱状图 |
| `W_Focus.pde` | **专注度检测**（~590 行）。基于 Alpha/Beta 比值计算专注度指标，用于神经反馈 |
| `W_EMG.pde` | **肌电信号显示**。EMG 专用显示模式，支持整流和包络检测 |
| `W_EMGJoystick.pde` | **肌电摇杆**（~660 行）。将 EMG 信号映射为 2D 摇杆位置，用于游戏控制 |
| `W_Accelerometer.pde` | **加速度计显示**。3 轴加速度计的实时 X/Y/Z 数值和方向可视化 |
| `W_AnalogRead.pde` | **模拟输入显示**（~510 行）。读取并显示辅助模拟引脚数据 |
| `W_DigitalRead.pde` | **数字输入显示**（~360 行）。读取并显示数字引脚状态 |
| `W_PulseSensor.pde` | **脉搏传感器**（~440 行）。心率信号实时显示和 BPM 计算 |
| `W_Networking.pde` | **网络数据流**（~2650 行）。通过 UDP/TCP 协议实时发送 EEG 数据到外部程序（如 Python 分析脚本） |
| `W_Playback.pde` | **回放控制面板**。数据回放模式下的播放/暂停/快进/倒退控制 |
| `W_PacketLoss.pde` | **丢包监控**。实时显示 WiFi 数据传输的丢包率 |
| `W_CytonImpedance.pde` | **阻抗检测**（~970 行）。Cyton 主板各通道电极阻抗测量与显示 |
| `W_GanglionImpedance.pde` | **Ganglion 阻抗检测**。Ganglion 主板的阻抗测量 |
| `W_Template.pde` | **Widget 模板**。新 Widget 开发的参考模板 |
| `Widget.pde` | **Widget 基类**（~660 行）。所有可视化组件的抽象基类，定义了统一的接口和生命周期 |
| `WidgetManager.pde` | **Widget 管理器**（~490 行）。管理所有 Widget 的创建、销毁、布局和拖拽 |

### 设置与配置模块

| 文件 | 功能 |
|:----|:----|
| `ControlPanel.pde` | **主控制面板**（~3500 行）。软件主界面，包含开始/停止采集、硬件设置、滤波设置、Widget 管理等所有功能的集成入口 |
| `SessionSettings.pde` | **Session 设置**（~1780 行）。数据记录路径、文件命名规则、Session 时长等实验参数配置 |
| `FilterSettings.pde` | **滤波设置**。滤波器类型、截止频率、阶数等参数配置 |
| `FilterUI.pde` | **滤波设置界面**（~1500 行）。滤波器参数的可视化配置面板 |
| `EmgSettings.pde` | **EMG 设置**。肌电信号处理的参数配置 |
| `EmgSettingsUI.pde` | **EMG 设置界面**（~630 行）。肌电参数的可视化配置面板 |
| `GuiSettings.pde` | **GUI 设置**。窗口大小、全屏模式、颜色主题等界面配置 |
| `RadioConfig.pde` | **无线配置**（~470 行）。WiFi 网络连接参数配置（SSID、密码、IP 地址等） |
| `TopNav.pde` | **顶部导航栏**（~1530 行）。程序顶部功能按钮栏（数据源选择、Widget 切换、设置入口） |
| `ADS1299SettingsBoard.pde` | **ADS1299 设置板接口**。ADC 芯片寄存器配置的抽象接口 |
| `ADS1299SettingsController.pde` | **ADS1299 设置控制器**。ADC 参数的读取/写入/验证 |

### 数据处理与辅助模块

| 文件 | 功能 |
|:----|:----|
| `DataProcessing.pde` | **信号处理核心**。在线滤波、FFT、包络检测等信号处理算法 |
| `ConsoleLog.pde` | **控制台日志**（~350 行）。程序运行日志的输出窗口 |
| `Debugging.pde` | **调试工具**。开发调试用的辅助功能函数 |
| `CytonElectrodeStatus.pde` | **电极状态检测**。Cyton 主板各通道的电连接状态实时检测 |
| `PacketLossTracker.pde` | **丢包追踪**。WiFi 数据传输中丢失数据包的统计与报告 |
| `TimeTrackingQueue.pde` | **时间追踪队列**。高精度时间戳记录，用于评估数据流的实时性 |
| `SignalCheckThresholds.pde` | **信号阈值检测**。EEG 信号幅度的异常检测（如电极脱落报警） |
| `FocusEnums.pde` | **专注度枚举定义**。专注度检测的状态机定义 |
| `FilterEnums.pde` | **滤波器枚举定义**。滤波器类型、窗口类型等枚举常量 |
| `EmgSettingsEnums.pde` | **EMG 枚举定义**。EMG 信号处理模式的枚举常量 |
| `ImpedanceSettingsBoard.pde` | **阻抗设置接口**。阻抗检测的硬件抽象接口 |
| `CytonImpedanceEnums.pde` | **阻抗枚举定义**。阻抗检测状态和模式的枚举常量 |

### 数据源与回放模块

| 文件 | 功能 |
|:----|:----|
| `DataSourcePlaybackCyton.pde` | **Cyton 数据回放**。回放 Cyton 格式的数据文件 |
| `DataSourcePlaybackGanglion.pde` | **Ganglion 数据回放**。回放 Ganglion 格式的数据文件 |
| `DataSourcePlaybackSynthetic.pde` | **合成数据回放**。生成模拟 EEG 信号用于测试 |
| `BoardBrainflow.pde` | **BrainFlow 板卡驱动**。兼容 BrainFlow 开源 EEG 框架 |
| `BoardBrainFlowStreaming.pde` | **BrainFlow 流式数据传输** |
| `BoardBrainFlowSynthetic.pde` | **BrainFlow 合成数据** |
| `TestPlaybackPanel.pde` | **回放测试面板**（~1250 行）。回放模式下的测试与调试界面 |

### 板卡抽象层

| 文件 | 功能 |
|:----|:----|
| `Board.pde` | **板卡基类**。所有 EEG 板卡的抽象基类，定义数据获取、采样率、通道数等统一接口 |
| `BoardCyton.pde` | **Cyton 主板驱动**。OpenBCI Cyton 的具体驱动实现（8/16 通道模式、增益切换） |
| `BoardGanglion.pde` | **Ganglion 主板驱动**。OpenBCI Ganglion 的具体驱动实现（4 通道） |
| `BoardNull.pde` | **空板卡**。无硬件时的占位实现，用于纯回放模式 |
| `FileBoard.pde` | **文件板卡**。从文件读取数据的板卡抽象 |
| `AnalogCapableBoard.pde` | **模拟引脚接口**。支持模拟输入的板卡接口 |
| `DigitalCapableBoard.pde` | **数字引脚接口**。支持数字 I/O 的板卡接口 |
| `AccelerometerCapableBoard.pde` | **加速度计接口**。支持加速度计读取的板卡接口 |
| `AuxDataBoard.pde` | **辅助数据接口**。额外数据通道的板卡接口 |
| `ImpedanceSettingsBoard.pde` | **阻抗设置接口**。支持阻抗测量配置的板卡接口 |
| `SmoothingBoard.pde` | **信号平滑接口**。支持硬件级信号平滑的板卡接口 |

### 其他文件

| 文件 | 功能 |
|:----|:----|
| `Buffer.pde` | **环形缓冲区**。高性能线程安全的数据缓冲区 |
| `Containers.pde` | **容器类**。自定义数据结构（数据点、数据帧等） |
| `CustomCp5Classes.pde` | **ControlP5 自定义类**。UI 控件库的自定义扩展 |
| `DirectoryManager.pde` | **目录管理器**。跨平台的存储路径管理 |
| `Extras.pde` | **附加功能**（~650 行）。杂项辅助函数 |
| `FixedStack.pde` | **固定大小栈**。容量固定的高效栈数据结构 |
| `GClip.pde` | **图形裁剪**。波形绘制的裁剪和缩放辅助 |
| `Grid.pde` | **网格布局**。Widget 的网格布局管理器 |
| `Interactivity.pde` | **交互功能**。拖拽、缩放、选择等用户交互处理 |
| `NotificationMessage.pde` | **通知消息**（~105 行）。顶部弹出式通知消息 |
| `eeg_viewer.html` | **Web 端 EEG 查看器**。基于 Web 的 EEG 数据查看器 |
| `AndroidManifest.xml` | **Android 清单**。Android 平台部署配置 |
| `Info.plist.tmpl` | **macOS 配置模板**。macOS 应用包配置文件 |
| `sketch.icns` | **macOS 图标**。应用图标文件 |
| `build.bat` / `run.bat` / `debug_build.bat` | **构建/运行脚本**。Windows 下的 Processing 编译和运行批处理脚本 |
| `README.md` | OpenBCI GUI 原始说明文档 |

---

## analysis/ — Python 分析脚本详解

`analysis/` 包含 28 个 Python 脚本，覆盖从原始数据加载到最终报告生成的全流程。

### 核心解码分析

| 脚本 | 功能 | 关键参数 |
|:----|:----|:----|
| `plot_region_comprehensive.py` | **15 脑区组合解码**。对 4 个单脑区 + 6 个两两组合 + 4 个三三组合 + 全部通道进行 4 分类 LDA 解码。同时评估两个维度：4 分类准确率和轴（垂直 vs 水平）准确率 | 15 组合 x 2 维度 x 5 折 CV |
| `plot_binary.py` | **二分类分析**。对 4 个方向的全部 6 组两两配对（C(4,2)）进行二分类 LDA，在 7 个关键脑区组合中筛选最强表现。额外计算轴判别（垂直 vs 水平）准确率 | 7 组合 x 7 配对 x 5 折 CV |
| `plot_binary_hierarchical.py` | **层次化解码**。将 4 分类拆解为"轴判别 → 轴内方向判别"的两阶段模型，定位解码瓶颈。附带频域特征对轴判别的改善分析 | 15 组合 x 层次化 + 时频对比 |
| `plot_frontal_binary.py` | **前额区详细二分类**。针对前额区（F3, Fz, F4），分析每个通道单独和组合的二分类表现（7 种通道组合 x 7 个方向配对） | 7 通道组合 x 7 配对 x 5 折 CV |

### 特征工程对比

| 脚本 | 功能 | 特征方案 |
|:----|:----|:----|
| `plot_region_pca.py` | **PCA 降维对比**。对 4 窗口均值特征进行 PCA（95% 方差保留），对比降维前后的 4 分类准确率 | 无 PCA vs PCA 95% |
| `plot_channel_with_range.py` | **峰-峰范围特征**。每个通道额外添加 4 个窗口的幅度范围作为特征（4 均值 + 4 范围 = 8 特征/通道），对比 4 特征基准 | 均值 vs 均值+范围 |
| `plot_full_timecourse.py` | **全时程 + PCA**。使用全部 500 个时间点（而非 4 个窗口均值）作为特征 + PCA 降维。对 12 个通道逐一对比 | 4 窗口 vs 500 点+PCA |
| `plot_shrinkage_lda.py` | **收缩 LDA**。在高维全时程特征上使用收缩 LDA（solver=lsqr, shrinkage=auto），避免协方差矩阵奇异 | 标准 LDA vs 收缩 LDA |
| `plot_timefreq.py` | **时频特征**。FFT 提取 5 个频带（Delta/Theta/Alpha/Beta/Gamma）的功率特征，对比窗口均值。附带单频带分析 | 窗口均值 vs 频带能量 vs 窗口频带+PCA |
| `plot_ovo.py` | **OvO 投票**。训练 6 个二分类器（一对一）通过投票决出最终类别，对比标准多分类 LDA | 标准 LDA vs OvO 投票 |

### 泛化与评估

| 脚本 | 功能 | 评估方法 |
|:----|:----|:----|
| `plot_cross_session.py` | **跨 Session 验证**。训练集：Sessions 1-4（777 trials），测试集：Session 5（197 trials）。与同 Session 5 折 CV 对照，评估模型泛化能力 | S1-4→S5 vs S2 内 CV |
| `plot_feature_importance.py` | **特征重要性**。基于跨 Session LDA 模型的 3 个判别函数系数绝对值均值，计算每个通道×窗口特征的重要性分数 | LDA 系数大小 |
| `plot_single_channel.py` | **单通道解码**。12 个通道各自用 4 窗口均值特征进行 4 分类解码，排序显示 | 12 通道 x 4 分类 |

### ERP 波形与源定位

| 脚本 | 功能 | 输出 |
|:----|:----|:----|
| `roi_activation.py` | **脑区 ERP 分析**。计算 4 个脑区的条件平均 ERP 波形，按方向比较各成分的幅度和潜伏期 | ERP 波形叠加图 |
| `erp_analysis.py` | **完整 ERP 分析**（~830 行）。包含跨 Session 平均 ERP、脑区激活时序、方向差异条形图等综合脑电信号分析 | 多图综合分析 |
| `plot_stimulus.py` | **刺激呈现分析**。从原始文件读取数据，可视化刺激时间和标记分布 | 刺激标记时序图 |
| `plot_4sessions_roi.py` | **4 Session 脑区对比**。分别绘制每个 Session 的 4 脑区 ERP 波形，评估 Session 间波形一致性 | 4 Session × 4 脑区波形 |
| `analyze_v2.py` | **综合分析 v2**（~680 行）。集成式分析脚本，涵盖数据加载、滤波、ERP 计算、激活热力图、时序分析等 | 多维度综合分析 |
| `plot_frontal_first10.py` | **前 10 试次分析**。对比前 10 试次与全部试次的差异，评估学习/疲劳效应 | 前 10 vs 全部 |
| `plot_roi_explanation.py` | **脑区解释图**。脑区划分示意图和通道-脑区对应关系的可视化说明 | 脑区分布说明图 |
| `plot_ica.py` | **ICA 分析**。独立成分分析，去除眼电/肌电等伪迹成分 | ICA 分解图 |
| `plot_sloreta.py` | **sLORETA 源定位**。基于 sLORETA 算法推算皮层电流源密度分布 | 皮层源定位图 |
| `plot_sloreta_all.py` | **sLORETA 跨 Session**。所有 Session 的源定位结果叠加对比 | 跨 Session 源定位 |
| `plot_sloreta_direction.py` | **sLORETA 方向分析**。按 4 方向分别计算皮层激活源 | 方向特异源定位 |
| `plot_sloreta_residual.py` | **sLORETA 残差分析**。分析头皮电位与源估计之间的残差分布 | 残差评估 |

### 报告生成

| 脚本 | 功能 |
|:----|:----|
| `generate_report.py` | **解码分析报告**。自动生成 13 章节的 Word 报告（.docx），包含所有解码实验的结果表格和图表 |
| `generate_technical_report.py` | **完整技术报告**。自动生成 7 章节的技术报告（.docx），包含算法原理、公式推导、代码示例和结果可视化 |

---

## reports/ — 分析报告

| 文件 | 内容 |
|:----|:----|
| `EEG_Arrow_Stimulus_Technical_Report.md` | **完整技术报告**（~570 行）。覆盖刺激程序、数据采集、原始数据格式、信号处理流程、模型算法及 Python 代码、15 脑区结果、6 组二分类、特征重要性、算法引用的全链路技术文档 |
| `binary_classification_report.md` | **二分类分析报告**。全部方向配对 × 脑区组合的二分类结果、层次化解码瓶颈分析、前额区详细混淆矩阵 |

---

## 环境依赖

### OpenBCI GUI（刺激呈现）
- **Processing 4.0+** (Java)
- **ControlP5** 库（UI 控件）
- OpenBCI Cyton / Cyton+Daisy 硬件 + WiFi Shield

### Python 分析
```
Python 3.10+
├── numpy, scipy          # 数值计算与信号处理
├── scikit-learn           # LDA 分类器, 交叉验证, 标准化
├── matplotlib             # 所有图表绘制
├── python-docx            # Word 报告自动生成
└── mne                    # sLORETA 源定位分析
```

---

## 运行方法

### 1. 启动刺激程序 + EEG 采集

```bash
# 在 Processing IDE 中打开
openbci/OpenBCI_GUI.pde

# 或通过命令行运行
cd openbci
./run.bat              # Windows
```

1. 连接 OpenBCI Cyton+Daisy 硬件
2. 在控制面板中配置通道增益、滤波参数
3. 点击 `Stimulus Launcher` 启动箭头刺激实验
4. 数据自动记录到 `Documents/OpenBCI_GUI/` 目录

### 2. 数据分析与解码

```bash
cd analysis

# 基础分析流程
python plot_region_comprehensive.py   # 15脑区解码
python plot_binary.py                 # 二分类分析
python plot_cross_session.py          # 跨Session泛化

# 特征工程对比
python plot_timefreq.py               # 时频特征
python plot_region_pca.py             # PCA降维
python plot_shrinkage_lda.py          # 收缩LDA

# 生成报告
python generate_technical_report.py   # 输出技术报告 .docx
python generate_report.py             # 输出解码报告 .docx
```

所有输出图片和报告保存到 `stimulus_logs/analysis_v5/` 目录。

---

## License

MIT License. OpenBCI GUI 部分基于 [OpenBCI_Processing](https://github.com/OpenBCI/OpenBCI_GUI) 二次开发，遵循其原始许可证。
