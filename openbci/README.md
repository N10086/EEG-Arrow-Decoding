# OpenBCI GUI - 视频刺激采集系统

基于 OpenBCI GUI v5.2.2 源码修改，用于 **8-16通道 WiFi EEG 设备** 的视觉刺激同步采集。

## 修改内容

- **StimulusController** — 本地视频文件刺激播放器，替代原有的 VEP/AEP 生成式刺激
  - 导入本地 MP4/AVI/MOV/MKV/WMV 视频文件作为视觉刺激
  - 使用系统默认播放器在独立窗口中播放（不遮挡主界面）
  - 支持窗口位置记忆
- **DataLogger 集成** — 视频播放标记自动写入 ODF 数据文件（Stimulus Marker 列）
- **同步控制** — Start Data Streaming 时自动播放视频，Stop Data Streaming 时自动停止（可选）
- **Processing 4.x 兼容性修复** — 解决预处理器关键字冲突、泛型解析等问题

## 使用说明

1. 编译并运行 GUI（见下方编译与运行）
2. 进入 Session 界面后，点击顶部导航栏的 **"导入视频"** 按钮
3. 在弹出的文件选择对话框中选中一个视频文件
4. 视频将自动在系统默认播放器中打开播放
5. 点击 **Start Data Stream** 开始 EEG 数据记录时，视频会同步播放
6. 点击 **Stop Data Stream** 停止记录时，视频停止
7. 可随时再次点击 **"导入视频"** 按钮更换其他视频文件
8. 视频播放窗口可以调整大小，关闭即停止播放

## 编译与运行

见 `build.bat`（编译）和 `run.bat`（运行），需先安装 Processing 4.5.2 并配置 BrainFlow 依赖。

### 编译
```
build.bat
```

### 运行
```
run.bat
```

## 文件说明

- `StimulusController.pde` — 视频刺激控制器与播放窗口
- `Interactivity.pde` — 键盘快捷键（无刺激相关快捷键）
- `TopNav.pde` — 顶部导航栏（含"导入视频"按钮）
