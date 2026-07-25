////////////////////////////////////////////////////////////////////////////////
//
//  StimulusController.pde  - 嵌入式视频刺激播放器
//
//  使用 JavaFX MediaPlayer 在控制窗口内直接播放视频，可精确读取进度条位置。
//  - 导入本地 MP4 文件
//  - 嵌入式播放：播放、暂停、拖动进度条
//  - 按 'k' 标记 → 记录视频进度条上的精确时间（毫秒级）
//  - 列表显示标记，可删除
//  - 在 ODF 文件中写入 Stimulus Marker 列
//
////////////////////////////////////////////////////////////////////////////////

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.ArrayList;
import java.util.concurrent.CopyOnWriteArrayList;

import javafx.stage.Stage;
import javafx.stage.StageStyle;
import javafx.embed.swing.JFXPanel;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.layout.StackPane;
import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import javafx.scene.media.MediaView;
import javafx.util.Duration;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;

// ★ 使用 Windows 系统外观，使 JFileChooser 等 Swing 组件显示原生风格
static {
    try {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
    } catch (Exception e) {
        // 非 Windows 平台不影响
    }
    // ★ JavaFX 运行时采用懒初始化（在第一次使用 Platform.runLater 前调用 ensureJavaFXToolkit()）
}

// ====================================================================
//  全局事件标记值
// ====================================================================
double stimulusMarkerValue = 0.0;
double pendingMarkerValue = -1.0;
String stimulusModeName = "No Video";

// ---- 刺激事件列表（供 W_TimeSeries 图表实时标注 + TXT 事后分析） ----
// 使用 CopyOnWriteArrayList 保证 JavaFX 线程（写入）与 Processing 线程（读取）安全
CopyOnWriteArrayList<StimulusEvent> stimulusEvents = new CopyOnWriteArrayList<StimulusEvent>();

// ---- 当前 session 目录（由 DataWriterODF 设置，CSV/PNG 共用） ----
String stimulusSessionDir = null;

class StimulusEvent {
    long wallTimeMs;    // 触发时的墙上时间 (ms)
    int markerNumber;   // 标记编号
    double value;       // 写入 ODF 的编码值 (2.0 + number/10000)
    String name;        // 可读描述（如 "↑ 上"、"↓ 下"），为空时 fallback 到 "M#N"

    StimulusEvent(long wallTimeMs, int markerNumber, double value) {
        this(wallTimeMs, markerNumber, value, "");
    }

    StimulusEvent(long wallTimeMs, int markerNumber, double value, String name) {
        this.wallTimeMs = wallTimeMs;
        this.markerNumber = markerNumber;
        this.value = value;
        this.name = (name != null) ? name : "";
    }
}

// ====================================================================
//  MarkerEntry  - 单个刺激标记
// ====================================================================
class MarkerEntry {
    int number;
    long videoTimeMs;      // 视频进度条上的精确时间（毫秒）
    long wallTimeMs;       // 墙上时钟（参考用）
    boolean deleted;

    MarkerEntry(int number, long videoTimeMs, long wallTimeMs) {
        this.number = number;
        this.videoTimeMs = videoTimeMs;
        this.wallTimeMs = wallTimeMs;
        this.deleted = false;
    }

    /** mm:ss.SSS 格式 */
    String getDisplay() {
        long totalSec = videoTimeMs / 1000;
        return String.format("%d:%02d.%03d", totalSec / 60, totalSec % 60, videoTimeMs % 1000);
    }
}

// ====================================================================
//  StimulusController  - 视频刺激主控制器
// ====================================================================
class StimulusController {

    private boolean isRunning = false;
    private String videoPath = "";
    private VideoPlaybackFrame videoFrame;
    private int winX = -1, winY = -1, winW = 640, winH = 480;

    // ---- 标记系统 ----
    private int markerCount = 0;
    private ArrayList<MarkerEntry> markerList;
    private PrintWriter markerLogWriter;
    private String markerLogPath;

    // ---- getter ----
    int getMarkerCount() { return markerCount; }
    ArrayList<MarkerEntry> getMarkers() { return markerList; }
    int getActiveMarkerCount() {
        if (markerList == null) return 0;
        int c = 0;
        for (MarkerEntry m : markerList) if (!m.deleted) c++;
        return c;
    }

    StimulusController() {
        markerList = new ArrayList<MarkerEntry>();
        String sx = System.getProperty("VIDEO_WIN_X");
        String sy = System.getProperty("VIDEO_WIN_Y");
        String sw = System.getProperty("VIDEO_WIN_W");
        String sh = System.getProperty("VIDEO_WIN_H");
        if (sx != null) try { winX = Integer.parseInt(sx); } catch (Exception e) {}
        if (sy != null) try { winY = Integer.parseInt(sy); } catch (Exception e) {}
        if (sw != null) try { winW = Integer.parseInt(sw); } catch (Exception e) {}
        if (sh != null) try { winH = Integer.parseInt(sh); } catch (Exception e) {}
    }

    void setVideoPath(String path) {
        this.videoPath = (path != null) ? path : "";
        stimulusModeName = (hasVideo()) ? new File(videoPath).getName() : "No Video";
    }

    String getVideoPath() { return videoPath; }
    boolean hasVideo() { return videoPath != null && !videoPath.isEmpty() && new File(videoPath).exists(); }

    boolean importVideo() {
        new Thread(new Runnable() {
            public void run() {
                File videoFile = showFileChooserBlocking("选择刺激视频文件", false, null);
                if (videoFile != null) {
                    final File f = videoFile;
                    SwingUtilities.invokeLater(new Runnable() {
                        public void run() {
                            onVideoSelected(f);
                        }
                    });
                }
            }
        }).start();
        return false;
    }

    // ---- 开始播放视频 ----
    void play() {
        if (!hasVideo()) {
            outputInfo("请先导入视频文件");
            return;
        }
        if (isRunning) stop();

        isRunning = true;
        stimulusMarkerValue = 1.0;
        markerCount = 0;
        markerList = new ArrayList<MarkerEntry>();
        pendingMarkerValue = -1.0;

        // 创建或显示播放窗口（内嵌视频播放）
        if (videoFrame == null || !videoFrame.isVisible()) {
            videoFrame = new VideoPlaybackFrame(videoPath, winX, winY, winW, winH);
        }

        outputInfo("视频刺激播放中: " + stimulusModeName);
        openMarkerLogFile();
    }

    // ---- 停止播放视频（静默清理，内部使用） ----
    void stop() {
        if (!isRunning) return;
        finishCleanup();
    }

    /** 用户主动关闭 — 若有标记则弹出保存对话框，否则直接清理 */
    void requestStop() {
        if (!isRunning) return;

        if (videoFrame != null) {
            videoFrame.saveWindowPosition();
        }

        if (getActiveMarkerCount() > 0) {
            int choice = showAlwaysOnTopConfirm(
                "有 " + getActiveMarkerCount() + " 个刺激标记点，是否保存？",
                "保存标记数据", JOptionPane.YES_NO_OPTION);
            if (choice != JOptionPane.YES_OPTION) {
                finishCleanup();
                return;
            }
            // 使用始终顶层的文件夹选择器
            new Thread(new Runnable() {
                public void run() {
                    File exportFolder = showFileChooserBlocking("选择保存标记数据的文件夹", true, null);
                    if (exportFolder != null) {
                        final File f = exportFolder;
                        SwingUtilities.invokeLater(new Runnable() {
                            public void run() {
                                onExportFolderSelected(f);
                            }
                        });
                    } else {
                        SwingUtilities.invokeLater(new Runnable() {
                            public void run() {
                                finishCleanup();
                            }
                        });
                    }
                }
            }).start();
            return;
        }

        finishCleanup();
    }

    /** 最终清理（无论是否导出都调用） */
    void finishCleanup() {
        isRunning = false;
        stimulusMarkerValue = 0.0;
        pendingMarkerValue = -1.0;

        if (videoFrame != null) {
            videoFrame.close();
            videoFrame = null;
        }
        closeMarkerLogFile();
        println("StimulusController: Video stopped. Markers: " + markerCount);
    }

    /** 弹出文件夹名称输入 + 是否复制MP4 对话框 */
    void promptExportOptions(String parentDir) {
        String videoName = (videoPath != null && !videoPath.isEmpty())
            ? new File(videoPath).getName() : "unknown";
        String baseName = videoName.replaceAll("\\.[^.]+$", "");
        String defaultName = "刺激标记_" + baseName;

        String folderName = showAlwaysOnTopInput("请输入文件夹名称：", "保存标记", defaultName);
        if (folderName == null || folderName.trim().isEmpty()) {
            finishCleanup();
            return;
        }
        folderName = folderName.trim();

        int copyChoice = showAlwaysOnTopConfirm(
            "是否将原始视频文件复制到导出文件夹？\n"
            + "（不复制则只保存标记 CSV，MP4 保留在原位置）",
            "复制视频文件", JOptionPane.YES_NO_OPTION);

        exportSession(parentDir, folderName, copyChoice == JOptionPane.YES_OPTION);
    }

    /** 导出标记数据 + 可选复制MP4到用户指定文件夹 */
    void exportSession(String parentDir, String folderName, boolean copyVideo) {
        if (markerList == null || markerList.isEmpty()) {
            finishCleanup();
            return;
        }

        // 创建目标文件夹
        File exportDir = new File(parentDir, folderName);
        if (!exportDir.exists()) {
            exportDir.mkdirs();
        }

        String videoName = (videoPath != null && !videoPath.isEmpty())
            ? new File(videoPath).getName() : "unknown";
        String baseName = videoName.replaceAll("\\.[^.]+$", "");
        String exportCsvPath = exportDir.getAbsolutePath() + File.separator
            + "刺激标记_" + baseName + ".csv";

        // 写入标记 CSV
        try {
            PrintWriter pw = new PrintWriter(new java.io.FileWriter(exportCsvPath));
            pw.println("# 刺激标记导出文件");
            pw.println("# 创建时间: " + new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(new Date()));
            pw.println("# 视频文件: " + (videoPath != null ? videoPath : ""));
            pw.println("# 标记总数: " + getActiveMarkerCount());
            pw.println("Marker#,VideoTime_ms,TimeDisplay");
            for (MarkerEntry entry : markerList) {
                if (!entry.deleted) {
                    pw.println(entry.number + "," + entry.videoTimeMs + "," + entry.getDisplay());
                }
            }
            pw.close();
        } catch (Exception e) {
            outputError("导出标记数据失败: " + e.getMessage());
            finishCleanup();
            return;
        }

        // 按需复制 MP4
        if (copyVideo && videoPath != null && !videoPath.isEmpty()) {
            try {
                java.nio.file.Path src = new File(videoPath).toPath();
                java.nio.file.Path dst = new File(exportDir, videoName).toPath();
                java.nio.file.Files.copy(src, dst,
                    java.nio.file.StandardCopyOption.COPY_ATTRIBUTES);
                outputInfo("视频文件已复制到导出文件夹");
            } catch (Exception e) {
                outputError("复制视频文件失败: " + e.getMessage());
            }
        }

        outputSuccess("标记数据已导出 → " + exportDir.getAbsolutePath());
        finishCleanup();
    }

    void update() {}

    boolean isActive() { return isRunning; }

    void saveFramePosition(int x, int y, int w, int h) {
        winX = x; winY = y; winW = w; winH = h;
        System.setProperty("VIDEO_WIN_X", String.valueOf(x));
        System.setProperty("VIDEO_WIN_Y", String.valueOf(y));
        System.setProperty("VIDEO_WIN_W", String.valueOf(w));
        System.setProperty("VIDEO_WIN_H", String.valueOf(h));
    }

    // ================================================================
    //  手动标记 — 记录视频进度条上的精确时间
    // ================================================================

    void mark() {
        if (!isRunning) return;

        // 从嵌入式播放器读取视频进度条上的精确时间
        long videoMs = (videoFrame != null) ? videoFrame.getCurrentVideoTimeMs() : 0;

        markerCount++;
        MarkerEntry entry = new MarkerEntry(markerCount, videoMs, System.currentTimeMillis());
        markerList.add(entry);

        pendingMarkerValue = 2.0 + markerCount / 10000.0;
        writeMarkerLog(entry);

        // 添加到刺激事件列表（供图表实时标注）
        stimulusEvents.add(new StimulusEvent(System.currentTimeMillis(), markerCount, pendingMarkerValue));

        if (videoFrame != null) {
            videoFrame.updateMarkerCount(getActiveMarkerCount());
            videoFrame.refreshMarkerList();
            videoFrame.writeSyncLogMarker(markerCount, entry.videoTimeMs);
        }

        println("StimulusController: Marker #" + markerCount + " @ " + entry.getDisplay());
        outputInfo("刺激标记 #" + markerCount + "  @" + entry.getDisplay());
    }

    void deleteMarker(int number) {
        for (MarkerEntry entry : markerList) {
            if (entry.number == number && !entry.deleted) {
                entry.deleted = true;
                if (markerLogWriter != null) {
                    try { markerLogWriter.println("#DELETED," + number + ",,,"); markerLogWriter.flush(); } catch (Exception e) {}
                }
                if (videoFrame != null) {
                    videoFrame.updateMarkerCount(getActiveMarkerCount());
                    videoFrame.refreshMarkerList();
                }
                return;
            }
        }
    }

    // ---- 标记日志 ----
    void openMarkerLogFile() {
        closeMarkerLogFile();
        try {
            markerLogPath = getStimulusLogDir() + "stimulus_markers_" + directoryManager.getFileNameDateTime() + ".csv";
            markerLogWriter = new PrintWriter(new java.io.FileWriter(markerLogPath));
            markerLogWriter.println("Marker#,VideoTime_ms,TimeDisplay,WallTime_ms,VideoFile,Status");
            markerLogWriter.flush();
        } catch (Exception e) { println("StimulusController: Cannot create marker log - " + e.getMessage()); }
    }

    private void writeMarkerLog(MarkerEntry entry) {
        if (markerLogWriter == null) return;
        try {
            markerLogWriter.println(entry.number + "," + entry.videoTimeMs + "," + entry.getDisplay()
                + "," + entry.wallTimeMs + "," + new File(videoPath).getName() + ",active");
            markerLogWriter.flush();
        } catch (Exception e) {}
    }

    void closeMarkerLogFile() {
        if (markerLogWriter == null) return;
        try {
            markerLogWriter.println("#END,,,,,");
            markerLogWriter.close();
        } catch (Exception e) {}
        markerLogWriter = null;
    }

    void resetMarkerCount() {
        markerCount = 0;
        markerList = new ArrayList<MarkerEntry>();
        pendingMarkerValue = -1.0;
    }
}


// ====================================================================
//  VideoPlaybackFrame  - 嵌入式视频播放控制窗口
//  使用 JavaFX MediaPlayer 直接在窗口内播放视频
// ====================================================================
class VideoPlaybackFrame {

    private JFrame frame;
    private MediaPlayer mediaPlayer;
    private String videoPath;

    // UI 控件（标记相关保留 Swing，播放控制移到 JavaFX Stage）
    private JButton markBtn;
    private JLabel markerCountLabel;
    private JPanel markerListPanel;
    private JScrollPane markerScrollPane;
    private JButton pinBtn;
    private boolean isPinned = false;

    // 拖动标记（防止拖动进度条时与播放器互相干扰）
    private volatile boolean seekingByUser = false;

    // 播放器状态
    private boolean playerReady = false;
    private long durationMs = 0;
    private final Object durationLock = new Object();

    // 倍速控制
    private double currentSpeed = 1.0;

    // 标签面板显示/隐藏
    private JPanel rootPanel;
    private JPanel topInfoPanel;
    private JPanel bottomInfoPanel;
    private JButton toggleMarkerBtn;
    private boolean markerPanelVisible = true;

    // ---- 自动标记触发（视频→EEG 同步） ----
    private long lastReportedPos = -1;
    private volatile boolean seekingOrJumping = false;
    private boolean firstPlayDone = false;

    // ---- 同步日志（~10ms 精度） ----
    private PrintWriter syncLogWriter;
    private ArrayList<String> syncLogBuffer;
    private javafx.animation.Timeline syncTimeline;

    // ---- 数据流同步 ----
    private boolean streamingState = false;
    private javax.swing.Timer streamCheckTimer;

    // ---- JavaFX Stage（独立窗口，替代 JFXPanel） ----
    private Stage videoStage;
    private MediaView mediaView;             // ✦ 常驻 MediaView（只创建一次，不随视频切换重建）
    private javafx.scene.control.Button fxPlayPauseBtn;
    private javafx.scene.control.Slider fxSeekSlider;
    private javafx.scene.control.Label fxTimeLabel;
    private javafx.scene.control.ComboBox<String> fxSpeedCombo;
    // ✦ 重试 & 日志
    private String currentVideoPath;
    private volatile int mediaRetryCount = 0;
    private static final int MAX_MEDIA_RETRIES = 2;

    VideoPlaybackFrame(String path, int x, int y, int w, int h) {
        this.videoPath = path;
        String fileName = new File(path).getName();

        frame = new JFrame("视频刺激 - " + fileName);
        frame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        frame.setResizable(true);
        frame.setAlwaysOnTop(false);
        frame.setFocusableWindowState(true);

        // ===== 主面板 =====
        rootPanel = new JPanel(new BorderLayout(5, 5));
        rootPanel.setBorder(BorderFactory.createEmptyBorder(8, 8, 5, 8));
        JPanel root = rootPanel; // local alias

        // ---- 顶部：文件名 + 标记计数 ----
        topInfoPanel = new JPanel(new BorderLayout(8, 3));
        JLabel infoLabel = new JLabel(fileName);
        infoLabel.setFont(new Font("Microsoft YaHei", Font.PLAIN, 12));
        topInfoPanel.add(infoLabel, BorderLayout.WEST);
        markerCountLabel = new JLabel("标记: 0", SwingConstants.RIGHT);
        markerCountLabel.setFont(new Font("Consolas", Font.BOLD, 14));
        markerCountLabel.setForeground(new Color(180, 0, 0));
        topInfoPanel.add(markerCountLabel, BorderLayout.EAST);
        root.add(topInfoPanel, BorderLayout.NORTH);

        // ---- 中央：标记列表 + 控制按钮 (视频在独立 JavaFX Stage 中) ----

        // ---- 底部：标记列表 + 按钮 ----
        bottomInfoPanel = new JPanel(new BorderLayout(5, 5));

        // 标记列表
        markerListPanel = new JPanel();
        markerListPanel.setLayout(new BoxLayout(markerListPanel, BoxLayout.Y_AXIS));
        markerScrollPane = new JScrollPane(markerListPanel);
        markerScrollPane.setBorder(BorderFactory.createTitledBorder("刺激标记列表"));
        markerScrollPane.setPreferredSize(new Dimension(360, 120));
        markerScrollPane.setMinimumSize(new Dimension(300, 40));
        bottomInfoPanel.add(markerScrollPane, BorderLayout.CENTER);

        // 按钮行
        JPanel buttonRow = new JPanel(new FlowLayout(FlowLayout.CENTER, 6, 3));

        markBtn = new JButton("★ 打标记 [k]");
        markBtn.setFont(new Font("Microsoft YaHei", Font.BOLD, 13));
        markBtn.setBackground(new Color(220, 50, 50));
        markBtn.setForeground(Color.WHITE);
        markBtn.setFocusPainted(false);
        markBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                if (stimController != null) stimController.mark();
            }
        });

        JButton selectBtn = new JButton("选择其他");
        selectBtn.setFont(new Font("Microsoft YaHei", Font.PLAIN, 12));
        selectBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                if (stimController != null) stimController.importVideo();
            }
        });

        pinBtn = new JButton("□ 置顶");
        pinBtn.setFont(new Font("Microsoft YaHei", Font.PLAIN, 12));
        pinBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) { togglePinned(); }
        });

        JButton fullScreenBtn = new JButton("▼ 隐藏标记");
        fullScreenBtn.setFont(new Font("Microsoft YaHei", Font.PLAIN, 12));
        fullScreenBtn.setFocusPainted(false);
        fullScreenBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) { toggleMarkerPanel(); }
        });
        toggleMarkerBtn = fullScreenBtn;

        JButton closeBtn = new JButton("关闭");
        closeBtn.setFont(new Font("Microsoft YaHei", Font.PLAIN, 12));
        closeBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                if (stimController != null) stimController.requestStop();
            }
        });

        buttonRow.add(markBtn);
        buttonRow.add(selectBtn);
        buttonRow.add(pinBtn);
        buttonRow.add(fullScreenBtn);
        buttonRow.add(closeBtn);
        bottomInfoPanel.add(buttonRow, BorderLayout.SOUTH);

        root.add(bottomInfoPanel, BorderLayout.SOUTH);
        frame.add(root);

        // 窗口位置和大小（仅标记列表，缩小尺寸）
        if (x >= 0 && y >= 0) frame.setLocation(x, y);
        else frame.setLocationRelativeTo(null);
        frame.setSize(380, 280);
        frame.setMinimumSize(new Dimension(300, 220));

        frame.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                saveWindowPosition();
                if (stimController != null) stimController.requestStop();
            }
            public void windowClosed(WindowEvent e) { saveWindowPosition(); }
        });

        // ---- 初始化 JavaFX 播放器（在窗口可见之前设置 Scene） ----
        Platform.setImplicitExit(false);
        initFXPlayer();

        frame.setVisible(true);

        // ---- 启动数据流状态监控（200ms 轮询） ----
        startStreamCheckTimer();
    }

    // ================================================================
    //  JavaFX 播放器初始化 — 常驻 Stage 架构
    //  Stage/Scene/MediaView 只创建一次 per instance，
    //  MediaPlayer 在切换视频或错误重试时替换（不重建 Stage）
    // ================================================================
    private void initFXPlayer() {
        final String fileName = new File(videoPath).getName();

        // ★ 确保 JavaFX 运行时已初始化
        ensureJavaFXToolkit();

        Platform.runLater(new Runnable() {
            public void run() {
                try {
                    println("[VideoPlaybackFrame] === 初始化 ===");

                    // 1. 创建常驻 Stage（仅首次执行，之后跳过）
                    createFXStage();

                    // 2. 加载当前视频
                    loadMedia(videoPath);

                } catch (Exception e) {
                    println("[VideoPlaybackFrame] initFXPlayer 异常: " + e.getMessage());
                    e.printStackTrace();
                    outputError("无法创建视频窗口: " + e.getMessage());
                }
            }
        });
    }

    // ----------------------------------------------------------------
    //  创建常驻 Stage + Scene + MediaView + 控制栏（per instance 只创建一次）
    // ----------------------------------------------------------------
    private void createFXStage() {
        if (videoStage != null) {
            println("[VideoPlaybackFrame] Stage 已存在，跳过创建");
            return;
        }

        String fileName = new File(videoPath).getName();
        println("[VideoPlaybackFrame] --- 创建常驻 Stage ---");

        videoStage = new Stage();
        videoStage.setTitle("视频 - " + fileName);
        videoStage.initStyle(StageStyle.DECORATED);
        videoStage.setAlwaysOnTop(false);

        // ★ 创建常驻 MediaView（不随视频切换重建）
        mediaView = new MediaView();
        mediaView.setPreserveRatio(true);

        // ---- JavaFX 控件 ----
        fxPlayPauseBtn = new javafx.scene.control.Button("▶ 播放");
        fxPlayPauseBtn.setOnAction(new javafx.event.EventHandler<javafx.event.ActionEvent>() {
            public void handle(javafx.event.ActionEvent e) {
                if (mediaPlayer == null) return;
                if (mediaPlayer.getStatus() == MediaPlayer.Status.PLAYING) {
                    mediaPlayer.pause();
                } else {
                    mediaPlayer.play();
                }
            }
        });

        fxSeekSlider = new javafx.scene.control.Slider(0, 1, 0);
        fxSeekSlider.setOnMousePressed(new javafx.event.EventHandler<javafx.scene.input.MouseEvent>() {
            public void handle(javafx.scene.input.MouseEvent e) { seekingByUser = true; }
        });
        fxSeekSlider.setOnMouseReleased(new javafx.event.EventHandler<javafx.scene.input.MouseEvent>() {
            public void handle(javafx.scene.input.MouseEvent e) {
                seekingByUser = false;
                if (mediaPlayer != null && playerReady) {
                    double pct = fxSeekSlider.getValue();
                    long target = (long)(durationMs * pct);
                    seekingOrJumping = true;
                    mediaPlayer.seek(Duration.millis(target));
                }
            }
        });

        fxTimeLabel = new javafx.scene.control.Label("0:00.000 / 0:00.000");

        fxSpeedCombo = new javafx.scene.control.ComboBox<>();
        fxSpeedCombo.getItems().addAll("0.10x","0.20x","0.30x","0.50x","0.75x","1.00x","1.25x","1.50x","2.00x");
        fxSpeedCombo.getSelectionModel().select(5);
        fxSpeedCombo.setOnAction(new javafx.event.EventHandler<javafx.event.ActionEvent>() {
            public void handle(javafx.event.ActionEvent e) {
                int idx = fxSpeedCombo.getSelectionModel().getSelectedIndex();
                double[] vals = {0.1,0.2,0.3,0.5,0.75,1.0,1.25,1.5,2.0};
                currentSpeed = (idx >= 0 && idx < vals.length) ? vals[idx] : 1.0;
                if (mediaPlayer != null) mediaPlayer.setRate(currentSpeed);
            }
        });

        // ---- 布局 ----
        javafx.scene.layout.BorderPane bpRoot = new javafx.scene.layout.BorderPane();
        bpRoot.setCenter(mediaView);

        javafx.scene.layout.HBox ctrlBar = new javafx.scene.layout.HBox(6);
        ctrlBar.setStyle("-fx-padding: 4 6 6 6; -fx-alignment: center-left;");
        fxPlayPauseBtn.setStyle("-fx-font-size: 12px;");
        fxSeekSlider.setPrefWidth(300);
        fxTimeLabel.setStyle("-fx-font-family: Consolas; -fx-font-size: 11px;");
        fxSpeedCombo.setStyle("-fx-font-family: Consolas; -fx-font-size: 11px;");
        ctrlBar.getChildren().addAll(fxPlayPauseBtn, fxSeekSlider, fxTimeLabel, fxSpeedCombo);
        bpRoot.setBottom(ctrlBar);

        Scene scene = new Scene(bpRoot, 640, 480);
        videoStage.setScene(scene);

        mediaView.fitWidthProperty().bind(bpRoot.widthProperty());
        mediaView.fitHeightProperty().bind(bpRoot.heightProperty().subtract(ctrlBar.heightProperty()));

        // ★ 窗口关闭时只暂停视频 + 隐藏（不销毁 Stage，可复用）
        videoStage.setOnCloseRequest(new javafx.event.EventHandler<javafx.stage.WindowEvent>() {
            public void handle(javafx.stage.WindowEvent e) {
                println("[VideoPlaybackFrame] Stage 关闭请求 → 暂停 + 隐藏");
                if (mediaPlayer != null) mediaPlayer.pause();
                videoStage.hide();
            }
        });

        // 窗口位置（紧贴 Swing 父窗口右侧）
        if (frame != null && frame.isVisible()) {
            Point loc = frame.getLocationOnScreen();
            videoStage.setX(loc.x + frame.getWidth() + 10);
            videoStage.setY(loc.y);
        }

        // ★ Stage 不立即 show()，等 onReady 触发后再显示
        println("[VideoPlaybackFrame] ✓ 常驻 Stage 创建完毕（未显示，等待 onReady）");
    }

    // ----------------------------------------------------------------
    //  加载视频（入口：切换视频 / 首次加载）
    // ----------------------------------------------------------------
    private void loadMedia(String path) {
        println("[VideoPlaybackFrame] --- loadMedia: " + path + " ---");
        currentVideoPath = path;
        mediaRetryCount = 0;
        doLoadMedia();
    }

    // ----------------------------------------------------------------
    //  执行 Media 加载 + 错误重试
    // ----------------------------------------------------------------
    private void doLoadMedia() {
        println("[VideoPlaybackFrame]    doLoadMedia [尝试 #" + (mediaRetryCount + 1)
            + "/" + (MAX_MEDIA_RETRIES + 1) + "]");

        // 1. 释放旧 MediaPlayer
        if (mediaPlayer != null) {
            try {
                println("[VideoPlaybackFrame]    释放旧 MediaPlayer...");
                mediaPlayer.stop();
                mediaPlayer.dispose();
            } catch (Exception e) {
                println("[VideoPlaybackFrame]    释放旧 MediaPlayer 异常: " + e.getMessage());
            }
            mediaPlayer = null;
        }

        // 2. 重置状态
        playerReady = false;
        lastReportedPos = -1;
        seekingOrJumping = false;
        firstPlayDone = false;
        synchronized (durationLock) {
            durationMs = 0;
        }

        try {
            // 3. 创建 Media
            String uri = new File(currentVideoPath).toURI().toString();
            println("[VideoPlaybackFrame]    URI: " + uri);
            Media media = new Media(uri);

            // ★ Media 级别错误监听（文件读取 / 格式识别失败）
            media.setOnError(new Runnable() {
                public void run() {
                    String errMsg = (media.getError() != null) ? media.getError().getMessage() : "未知";
                    println("[VideoPlaybackFrame]    ✗ [Media.onError] " + errMsg);
                }
            });

            // 4. 创建新 MediaPlayer
            MediaPlayer newPlayer = new MediaPlayer(media);
            newPlayer.setRate(currentSpeed);
            println("[VideoPlaybackFrame]    MediaPlayer 已创建");

            // ---- 初始化同步日志时间线 ----
            startSyncTimeline();

            // ★ 当前时间监听（进度条 + 时间显示）
            newPlayer.currentTimeProperty().addListener(new javafx.beans.value.ChangeListener<Duration>() {
                public void changed(javafx.beans.value.ObservableValue<? extends Duration> obs, Duration old, Duration now) {
                    if (!seekingByUser && playerReady) {
                        final long cur = (long) now.toMillis();
                        synchronized (durationLock) {
                            if (durationMs > 0) {
                                double pct = (double) cur / durationMs;
                                fxSeekSlider.setValue(pct);
                                fxTimeLabel.setText(formatTimeFX(cur) + " / " + formatTimeFX(durationMs));
                            }
                        }
                    }
                }
            });

            // ★ 状态监听（播放/暂停 按钮文字 + 详细日志）
            newPlayer.statusProperty().addListener(new javafx.beans.value.ChangeListener<MediaPlayer.Status>() {
                public void changed(javafx.beans.value.ObservableValue<? extends MediaPlayer.Status> obs, MediaPlayer.Status old, MediaPlayer.Status stat) {
                    fxPlayPauseBtn.setText(stat == MediaPlayer.Status.PLAYING ? "⏸ 暂停" : "▶ 播放");
                    if (stat == MediaPlayer.Status.PLAYING) {
                        println("[VideoPlaybackFrame] ▶ [status=PLAYING]");
                    } else if (stat == MediaPlayer.Status.PAUSED) {
                        println("[VideoPlaybackFrame] ⏸ [status=PAUSED]");
                    } else if (stat == MediaPlayer.Status.STOPPED) {
                        println("[VideoPlaybackFrame] ⏹ [status=STOPPED]");
                    } else if (stat == MediaPlayer.Status.READY) {
                        println("[VideoPlaybackFrame] ✓ [status=READY]");
                    }
                }
            });

            // 5. ★ onReady — 视频就绪后才显示 Stage
            newPlayer.setOnReady(new Runnable() {
                public void run() {
                    synchronized (durationLock) {
                        durationMs = (long) newPlayer.getTotalDuration().toMillis();
                    }
                    playerReady = true;
                    mediaRetryCount = 0;  // 成功后重置计数
                    println("[VideoPlaybackFrame] ✓ [onReady] 视频已就绪, 时长=" + durationMs + "ms");

                    // ★ 只有 onReady 成功后才显示 Stage（避免空白/黑屏）
                    if (videoStage != null && !videoStage.isShowing()) {
                        println("[VideoPlaybackFrame] → 首次显示 Stage (onReady 后)");
                        videoStage.show();
                    }
                }
            });

            // 6. ★★★ onError — 错误处理 + 自动重试 ★★★
            newPlayer.setOnError(new Runnable() {
                public void run() {
                    String errMsg = "未知";
                    try {
                        Throwable err = newPlayer.getError();
                        if (err != null) {
                            errMsg = err.getMessage();
                            if (errMsg == null) errMsg = err.getClass().getName();
                            println("[VideoPlaybackFrame] 错误详情: " + errMsg);
                        }
                    } catch (Exception ex) {
                        errMsg = ex.getMessage();
                        if (errMsg == null) errMsg = "未知异常";
                    }
                    println("[VideoPlaybackFrame] ⚠ [onError] " + errMsg);

                    if (mediaRetryCount < MAX_MEDIA_RETRIES) {
                        mediaRetryCount++;
                        println("[VideoPlaybackFrame] ⚠ 500ms 后自动重试第 " + mediaRetryCount + " 次...");
                        javafx.animation.PauseTransition pt = new javafx.animation.PauseTransition(Duration.millis(500));
                        pt.setOnFinished(new javafx.event.EventHandler<javafx.event.ActionEvent>() {
                            public void handle(javafx.event.ActionEvent e) {
                                doLoadMedia();
                            }
                        });
                        pt.play();
                    } else {
                        println("[VideoPlaybackFrame] ✗ 已达最大重试次数 (" + MAX_MEDIA_RETRIES + ")，放弃");
                        outputError("视频播放失败（已重试 " + MAX_MEDIA_RETRIES + " 次）: " + errMsg);
                        // 兜底：即便失败也显示 Stage
                        if (videoStage != null && !videoStage.isShowing()) {
                            println("[VideoPlaybackFrame] → 兜底显示 Stage（媒体加载失败）");
                            videoStage.show();
                        }
                    }
                }
            });

            // 7. 绑定到常驻 MediaView
            mediaView.setMediaPlayer(newPlayer);
            mediaPlayer = newPlayer;

            println("[VideoPlaybackFrame]    MediaPlayer 绑定完毕，等待 onReady...");

        } catch (Exception e) {
            println("[VideoPlaybackFrame]    ✗ doLoadMedia 异常: " + e.getMessage());
            e.printStackTrace();
            if (mediaRetryCount < MAX_MEDIA_RETRIES) {
                mediaRetryCount++;
                println("[VideoPlaybackFrame] ⚠ 500ms 后异常恢复重试第 " + mediaRetryCount + " 次...");
                javafx.animation.PauseTransition pt = new javafx.animation.PauseTransition(Duration.millis(500));
                pt.setOnFinished(new javafx.event.EventHandler<javafx.event.ActionEvent>() {
                    public void handle(javafx.event.ActionEvent e) {
                        doLoadMedia();
                    }
                });
                pt.play();
            } else {
                outputError("视频加载异常（已重试 " + MAX_MEDIA_RETRIES + " 次）: " + e.getMessage());
            }
        }
    }

    /** 获取视频进度条上的精确时间（毫秒）— 由 mark() 调用 */
    long getCurrentVideoTimeMs() {
        if (mediaPlayer == null || !playerReady) return 0;
        try {
            return (long) mediaPlayer.getCurrentTime().toMillis();
        } catch (Exception e) {
            return 0;
        }
    }

    // ---- 置顶 ----
    void togglePinned() {
        isPinned = !isPinned;
        frame.setAlwaysOnTop(isPinned);
        // 同步设置视频 Stage 的置顶状态
        if (videoStage != null) {
            Platform.runLater(new Runnable() {
                public void run() {
                    videoStage.setAlwaysOnTop(isPinned);
                }
            });
        }
        pinBtn.setText(isPinned ? "■ 置顶" : "□ 置顶");
        pinBtn.setFont(new Font("Microsoft YaHei", isPinned ? Font.BOLD : Font.PLAIN, 12));
    }

    // ---- 显示/隐藏底部标签面板 ----
    private void toggleMarkerPanel() {
        markerPanelVisible = !markerPanelVisible;
        if (markerScrollPane != null) {
            markerScrollPane.setVisible(markerPanelVisible);
            bottomInfoPanel.revalidate();
            bottomInfoPanel.repaint();
        }
        toggleMarkerBtn.setText(markerPanelVisible ? "▼ 隐藏标记" : "▲ 显示标记");
    }

    // ---- 标记列表刷新 ----
    void refreshMarkerList() {
        if (markerListPanel == null || stimController == null) return;
        markerListPanel.removeAll();
        ArrayList<MarkerEntry> markers = stimController.getMarkers();

        if (markers == null || markers.isEmpty()) {
            markerListPanel.add(new JLabel("  暂无标记"));
        } else {
            for (int i = markers.size() - 1; i >= 0; i--) {
                final MarkerEntry entry = markers.get(i);
                JPanel row = new JPanel(new BorderLayout(6, 2));
                row.setBorder(BorderFactory.createEmptyBorder(2, 4, 2, 4));

                JLabel lab = new JLabel(" #" + entry.number + "  " + entry.getDisplay());
                lab.setFont(new Font("Consolas", Font.PLAIN, 13));

                if (entry.deleted) {
                    lab.setForeground(Color.GRAY);
                    lab.setText("<html><strike>#" + entry.number + "  " + entry.getDisplay() + "</strike></html>");
                    row.add(lab, BorderLayout.CENTER);
                } else {
                    lab.setForeground(Color.BLACK);
                    row.add(lab, BorderLayout.CENTER);
                    JButton delBtn = new JButton("删除");
                    delBtn.setFont(new Font("Microsoft YaHei", Font.PLAIN, 11));
                    delBtn.setBackground(new Color(240, 200, 200));
                    delBtn.setFocusPainted(false);
                    delBtn.addActionListener(new ActionListener() {
                        public void actionPerformed(ActionEvent e) {
                            if (stimController != null) stimController.deleteMarker(entry.number);
                        }
                    });
                    row.add(delBtn, BorderLayout.EAST);
                }
                markerListPanel.add(row);
                if (i > 0) {
                    JSeparator sep = new JSeparator(SwingConstants.HORIZONTAL);
                    sep.setMaximumSize(new Dimension(32767, 1));
                    markerListPanel.add(sep);
                }
            }
        }
        markerListPanel.revalidate();
        markerListPanel.repaint();
    }

    void saveWindowPosition() {
        if (frame != null && stimController != null) {
            stimController.saveFramePosition(frame.getLocation().x, frame.getLocation().y,
                frame.getSize().width, frame.getSize().height);
        }
    }

    void updateMarkerCount(int count) {
        if (markerCountLabel != null) markerCountLabel.setText("标记: " + count);
        if (markBtn != null) {
            markBtn.setBackground(new Color(255, 100, 100));
            javax.swing.Timer t = new javax.swing.Timer(150, new ActionListener() {
                public void actionPerformed(ActionEvent e) { markBtn.setBackground(new Color(220, 50, 50)); }
            });
            t.setRepeats(false);
            t.start();
        }
    }

    boolean isVisible() { return frame != null && frame.isVisible(); }

    private String formatTimeFX(long ms) {
        long s = ms / 1000;
        return String.format("%d:%02d.%03d", s / 60, s % 60, ms % 1000);
    }

    void close() {
        println("[VideoPlaybackFrame] === close() ===");
        // 停止同步日志和时间线
        stopSyncLog();
        stopSyncTimeline();
        if (streamCheckTimer != null) {
            streamCheckTimer.stop();
            streamCheckTimer = null;
        }

        if (mediaPlayer != null) {
            println("[VideoPlaybackFrame] 停止 + 释放 MediaPlayer...");
            final MediaPlayer mp = mediaPlayer;
            mediaPlayer = null;
            final java.util.concurrent.CountDownLatch latch =
                new java.util.concurrent.CountDownLatch(1);
            Platform.runLater(new Runnable() {
                public void run() {
                    try {
                        // 关闭常驻 Stage（释放原生窗口资源）
                        if (videoStage != null) {
                            println("[VideoPlaybackFrame] 关闭 Stage...");
                            videoStage.close();
                            videoStage = null;
                        }
                        println("[VideoPlaybackFrame] dispose MediaPlayer...");
                        mp.stop();
                        mp.dispose();
                        println("[VideoPlaybackFrame] ✓ MediaPlayer 已释放");
                    } catch (Exception e) {
                        println("[VideoPlaybackFrame] dispose error - " + e.getMessage());
                    } finally {
                        latch.countDown();
                    }
                }
            });
            try {
                latch.await(3, java.util.concurrent.TimeUnit.SECONDS);
                println("[VideoPlaybackFrame] ✓ dispose 完成");
            } catch (InterruptedException ie) {
                println("[VideoPlaybackFrame] await 被中断");
            }
        } else {
            // 即使没有 MediaPlayer，也要清理 Stage
            if (videoStage != null) {
                println("[VideoPlaybackFrame] 关闭 Stage (no MediaPlayer)...");
                final Stage vs = videoStage;
                videoStage = null;
                Platform.runLater(new Runnable() {
                    public void run() { vs.close(); }
                });
            }
        }
        if (frame != null) {
            println("[VideoPlaybackFrame] 关闭 Swing Frame...");
            frame.dispose();
            frame = null;
        }
        println("[VideoPlaybackFrame] === close() 完成 ===");
    }

    // ================================================================
    //  数据流状态监控
    // ================================================================
    private void startStreamCheckTimer() {
        streamCheckTimer = new javax.swing.Timer(200, new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                boolean boardStreaming = (currentBoard != null && currentBoard.isStreaming());
                if (boardStreaming && !streamingState) {
                    streamingState = true;
                    startSyncLog();
                    Platform.runLater(new Runnable() {
                        public void run() {
                            if (mediaPlayer != null) {
                                outputInfo("数据流已启动，自动播放视频");
                                mediaPlayer.play();
                            }
                        }
                    });
                } else if (!boardStreaming && streamingState) {
                    streamingState = false;
                    stopSyncLog();
                    Platform.runLater(new Runnable() {
                        public void run() {
                            if (mediaPlayer != null) {
                                outputInfo("数据流已停止，暂停视频");
                                mediaPlayer.pause();
                            }
                        }
                    });
                }
            }
        });
        streamCheckTimer.start();
    }

    // ================================================================
    //  同步日志（10ms 精度）
    // ================================================================
    private void startSyncLog() {
        if (syncLogWriter != null) return;
        try {
            String logPath = getStimulusLogDir() + "sync_log_" + directoryManager.getFileNameDateTime() + ".csv";
            syncLogWriter = new PrintWriter(new java.io.FileWriter(logPath));
            syncLogWriter.println("# Sync log for video-EEG synchronization");
            syncLogWriter.println("# WallTime_ms,VideoPosition_ms,IsPlaying,MarkerFired");
            syncLogBuffer = new ArrayList<String>();
            println("SyncLog: 已创建 -> " + logPath);
        } catch (Exception e) {
            println("SyncLog: 创建失败 - " + e.getMessage());
        }
    }

    private void startSyncTimeline() {
        if (syncTimeline != null) return;
        syncTimeline = new Timeline();
        syncTimeline.setCycleCount(Timeline.INDEFINITE);
        syncTimeline.getKeyFrames().add(new KeyFrame(Duration.millis(10), new javafx.event.EventHandler<javafx.event.ActionEvent>() {
            public void handle(javafx.event.ActionEvent event) {
                if (!playerReady) return;
                long cur = (long) mediaPlayer.getCurrentTime().toMillis();

                // Seek 检测：位置稳定后清除 seekingOrJumping 标志
                if (seekingOrJumping) {
                    long delta = Math.abs(cur - lastReportedPos);
                    if (delta < 100 && lastReportedPos >= 0) {
                        seekingOrJumping = false;
                    }
                    lastReportedPos = cur;
                }

                // 写入同步日志（仅数据流活跃时）
                if (syncLogWriter != null) {
                    boolean playing = mediaPlayer.getStatus() == MediaPlayer.Status.PLAYING;
                    syncLogBuffer.add(System.currentTimeMillis() + "," + cur + "," + (playing ? "1" : "0") + ",");
                    if (syncLogBuffer.size() >= 100) flushSyncLog();
                }
            }
        }));
        syncTimeline.play();
    }

    void writeSyncLogMarker(int markerNumber, long videoMs) {
        if (syncLogWriter == null && !streamingState) return;
        // 如果日志未创建但数据流已激活，立即创建
        if (syncLogWriter == null && streamingState) startSyncLog();
        if (syncLogWriter == null) return;
        syncLogBuffer.add(System.currentTimeMillis() + "," + videoMs + ",1," + markerNumber);
        if (syncLogBuffer.size() >= 100) flushSyncLog();
    }

    private void flushSyncLog() {
        if (syncLogWriter == null || syncLogBuffer == null || syncLogBuffer.isEmpty()) return;
        try {
            for (String line : syncLogBuffer) {
                syncLogWriter.println(line);
            }
            syncLogWriter.flush();
            syncLogBuffer.clear();
        } catch (Exception e) {
            println("SyncLog: 写入失败 - " + e.getMessage());
        }
    }

    private void stopSyncLog() {
        flushSyncLog();
        if (syncLogWriter != null) {
            try {
                syncLogWriter.println("# END");
                syncLogWriter.close();
            } catch (Exception e) { }
            syncLogWriter = null;
        }
    }

    private void stopSyncTimeline() {
        if (syncTimeline != null) {
            syncTimeline.stop();
            syncTimeline = null;
        }
    }
}

// ====================================================================
//  全局辅助：永远在顶层的对话框
// ====================================================================
/** 显示永远在顶层的确认对话框 */
static int showAlwaysOnTopConfirm(String message, String title, int optionType) {
    JOptionPane pane = new JOptionPane(message, JOptionPane.QUESTION_MESSAGE, optionType);
    JDialog dialog = pane.createDialog(null, title);
    dialog.setAlwaysOnTop(true);
    dialog.setVisible(true);
    Object val = pane.getValue();
    return (val instanceof Integer) ? (Integer) val : JOptionPane.CLOSED_OPTION;
}

/** 显示永远在顶层的消息对话框 */
static void showAlwaysOnTopMessage(String message, String title, int messageType) {
    JOptionPane pane = new JOptionPane(message, messageType);
    JDialog dialog = pane.createDialog(null, title);
    dialog.setAlwaysOnTop(true);
    dialog.setVisible(true);
}

/** 显示永远在顶层的输入对话框，返回用户输入的文本，null 表示取消 */
static String showAlwaysOnTopInput(String message, String title, String initialValue) {
    JPanel panel = new JPanel(new BorderLayout(10, 10));
    JLabel label = new JLabel(message);
    JTextField textField = new JTextField(initialValue, 20);
    panel.add(label, BorderLayout.NORTH);
    panel.add(textField, BorderLayout.CENTER);
    JOptionPane pane = new JOptionPane(panel, JOptionPane.PLAIN_MESSAGE, JOptionPane.OK_CANCEL_OPTION);
    JDialog dialog = pane.createDialog(null, title);
    dialog.setAlwaysOnTop(true);
    dialog.setVisible(true);
    Object val = pane.getValue();
    if (val instanceof Integer && (Integer) val == JOptionPane.OK_OPTION) {
        return textField.getText();
    }
    return null;
}

/**
 * 显示永远在顶层的文件/文件夹选择对话框（阻塞调用）
 *
 * ● 选择文件 → 使用 AWT FileDialog（Windows 原生对话框，美观）
 * ● 选择文件夹 → 使用 Swing JFileChooser（始终置顶 JDialog 包装）
 *
 * @param title 标题
 * @param dirMode true=文件夹选择, false=文件选择
 * @param defaultDir 默认目录
 * @return 选中的文件，取消则 null
 */
static File showFileChooserBlocking(String title, boolean dirMode, File defaultDir) {
    if (!dirMode) {
        // ★ 文件选择：使用 AWT FileDialog（Windows/11 原生文件管理器）
        java.awt.Frame[] frames = java.awt.Frame.getFrames();
        java.awt.Frame parent = null;
        for (java.awt.Frame f : frames) {
            if (f.isVisible()) { parent = f; break; }
        }
        java.awt.FileDialog fd = new java.awt.FileDialog(parent, title, java.awt.FileDialog.LOAD);
        if (defaultDir != null && defaultDir.isDirectory()) {
            fd.setDirectory(defaultDir.getAbsolutePath());
        }
        // 设置文件过滤器（只显示所有文件）
        fd.setFilenameFilter(new java.io.FilenameFilter() {
            public boolean accept(java.io.File dir, String name) { return true; }
        });
        fd.setVisible(true);
        String dir = fd.getDirectory();
        String file = fd.getFile();
        if (dir != null && file != null) {
            return new java.io.File(dir, file);
        }
        return null;
    }

    // ★ 文件夹选择：Swing JFileChooser 包装在始终顶层的对话框中
    //    Java 没有原生的文件夹选择对话框 API，但 JFileChooser
    //    配合 Windows LAF 后外观接近原生。
    //    如果用户启用了 Windows LAF，JFileChooser 自动使用系统主题。
    final JFileChooser chooser = new JFileChooser();
    chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
    chooser.setDialogTitle(title);
    if (defaultDir != null) {
        chooser.setCurrentDirectory(defaultDir);
    }

    final JDialog dialog = new JDialog();
    dialog.setAlwaysOnTop(true);
    dialog.setModal(true);
    dialog.setTitle(title);
    dialog.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);

    JPanel panel = new JPanel(new BorderLayout());
    panel.add(chooser, BorderLayout.CENTER);
    dialog.getContentPane().add(panel);
    dialog.pack();
    dialog.setSize(Math.max(600, dialog.getWidth()), Math.max(400, dialog.getHeight()));
    dialog.setLocationRelativeTo(null);

    final File[] result = new File[1];
    chooser.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent e) {
            if (JFileChooser.APPROVE_SELECTION.equals(e.getActionCommand())) {
                result[0] = chooser.getSelectedFile();
            }
            dialog.dispose();
        }
    });

    dialog.setVisible(true);
    return result[0];
}

// 测试回放面板单例引用（由 onTestFolderSelected / onReloadFolderSelected 管理）
TestPlaybackPanel currentTestPanel = null;

// ====================================================================
//  selectFolder() 回调 — 测试模式：选择标记文件夹（首次加载）
// ====================================================================
void onTestFolderSelected(File folder) {
    if (folder == null) return;

    // ★ 停止可能仍在运行的 StimulusController（之前视频标记未关闭）
    if (stimController != null && stimController.isActive()) {
        println("TestPlaybackPanel: 停止正在运行的 StimulusController");
        stimController.finishCleanup();
    }

    // ★ 清除视频路径，防止 OpenBCI_GUI.pde:820 在 start data stream 时
    //   自动调用 stimController.play() 创建 VideoPlaybackFrame 窗口
    if (stimController != null) {
        stimController.setVideoPath(null);
    }

    // ★ 清除旧的刺激事件，避免之前测试/标记的标记点堆积在图表中
    stimulusEvents.clear();

    try {
        if (currentTestPanel != null) {
            // 如果已有面板且尚未 disposed，复用
            if (currentTestPanel.isDisposed()) {
                currentTestPanel = null;
                currentTestPanel = new TestPlaybackPanel(folder.getAbsolutePath());
            } else {
                currentTestPanel.loadFolder(folder.getAbsolutePath());
            }
        } else {
            currentTestPanel = new TestPlaybackPanel(folder.getAbsolutePath());
        }
    } catch (Exception e) {
        // 如果构造失败，确保不会有幽灵窗口残留
        // （现在 createGUI 不显示窗口，loadFolder 验证通过后才显示）
        outputError("测试回放失败: " + e.getMessage());
        println("TestPlaybackPanel: " + e.getMessage());
    }
}

// ====================================================================
//  selectFolder() 回调 — 测试模式：切换文件夹（复用面板）
// ====================================================================
void onReloadFolderSelected(File folder) {
    if (folder == null || currentTestPanel == null) return;
    try {
        currentTestPanel.loadFolder(folder.getAbsolutePath());
    } catch (Exception e) {
        outputError("测试回放失败: " + e.getMessage());
        println("TestPlaybackPanel: " + e.getMessage());
    }
}
// ====================================================================
//  selectFolder() 回调 — 关闭视频时导出标记数据
// ====================================================================
void onExportFolderSelected(File folder) {
    if (stimController == null) return;
    if (folder == null) {
        stimController.finishCleanup();
        return;
    }
    stimController.promptExportOptions(folder.getAbsolutePath());
}

// ====================================================================
//  selectInput() 回调
// ====================================================================
void onVideoSelected(File selection) {
    if (selection != null && selection.exists() && selection.isFile()) {
        String p = selection.getAbsolutePath().toLowerCase();
        if (p.endsWith(".mp4") || p.endsWith(".avi") || p.endsWith(".mov")
            || p.endsWith(".mkv") || p.endsWith(".wmv") || p.endsWith(".flv") || p.endsWith(".webm")) {
            if (stimController != null) {
                stimController.setVideoPath(selection.getAbsolutePath());
                stimController.play();
            }
        } else {
            outputError("不支持的文件格式");
        }
    }
}

// ====================================================================
//  selectInput() 回调 — 数据分析：选择 ODF .txt 文件 → 调用 Python 脚本生成图表
// ====================================================================
void onAnalysisFileSelected(File selection) {
    if (selection == null || !selection.exists()) return;

    final String txtPath = selection.getAbsolutePath();

    // 校验扩展名
    if (!txtPath.toLowerCase().endsWith(".txt")) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                showAlwaysOnTopMessage("请选择 ODF 格式的 .txt 文件",
                    "文件格式错误", JOptionPane.WARNING_MESSAGE);
            }
        });
        return;
    }

    outputInfo("正在分析数据: " + selection.getName());

    // 在后台线程中执行 Python 脚本，避免阻塞 UI
    new Thread(new Runnable() {
        public void run() {
            runAnalysis(txtPath);
        }
    }).start();
}

/** 在后台线程中执行分析脚本 */
void runAnalysis(String txtPath) {
    try {
        // ── 1. 定位 plot_stimulus.py ──
        File scriptFile = findPlotScriptFile();
        if (scriptFile == null) {
            SwingUtilities.invokeLater(new Runnable() {
                public void run() {
                    String msg = "找不到 plot_stimulus.py 脚本文件\n"
                        + "请确保该文件位于程序主目录下";
                    showAlwaysOnTopMessage(msg, "脚本未找到", JOptionPane.ERROR_MESSAGE);
                }
            });
            return;
        }

        // ── 2. 定位 Python 解释器 ──
        String python = findPython();
        if (python == null) {
            SwingUtilities.invokeLater(new Runnable() {
                public void run() {
                    String msg = "找不到 Python 解释器\n"
                        + "请安装 Python 3 并确保已添加到系统 PATH 环境变量";
                    showAlwaysOnTopMessage(msg, "Python 未找到", JOptionPane.ERROR_MESSAGE);
                }
            });
            return;
        }

        // ── 3. 执行 Python 脚本 ──
        ProcessBuilder pb = new ProcessBuilder(
            python,
            scriptFile.getAbsolutePath(),
            txtPath
        );
        pb.redirectErrorStream(true);
        Process p = pb.start();

        // 读取输出
        BufferedReader reader = new BufferedReader(
            new InputStreamReader(p.getInputStream(), "UTF-8"));
        StringBuilder output = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            output.append(line).append('\n');
        }
        int exitCode = p.waitFor();
        final String scriptOutput = output.toString();

        // ── 4. 处理结果 ──
        final String outPng = txtPath.replaceAll("(?i)\\.txt$", "_stimulus.png");
        final boolean success = exitCode == 0 && new File(outPng).exists();

        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                if (success) {
                    outputSuccess("刺激标注图表已生成: " + outPng);

                    int choice = showAlwaysOnTopConfirm(
                        "刺激标注图表已生成完毕\n\n"
                        + "输出文件: " + outPng + "\n\n"
                        + "是否立即打开查看？",
                        "分析完成",
                        JOptionPane.YES_NO_OPTION);

                    if (choice == JOptionPane.YES_OPTION) {
                        openFile(outPng);
                    }
                } else {
                    outputError("图表生成失败 (exit=" + exitCode + ")");
                    showAlwaysOnTopMessage("图表生成失败，请检查控制台输出\n\n"
                        + scriptOutput,
                        "分析失败", JOptionPane.ERROR_MESSAGE);
                }
            }
        });

    } catch (Exception e) {
        final String errMsg = e.getMessage();
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                outputError("运行分析脚本异常: " + errMsg);
                showAlwaysOnTopMessage("运行分析脚本时发生异常:\n" + errMsg,
                    "错误", JOptionPane.ERROR_MESSAGE);
            }
        });
    }
}

/** 查找 plot_stimulus.py 脚本文件 */
File findPlotScriptFile() {
    // 尝试方式1: sketchPath (Processing PDE / 打包应用)
    try {
        String sp = sketchPath("plot_stimulus.py");
        if (sp != null) {
            File f = new File(sp);
            if (f.exists()) return f;
        }
    } catch (Exception e) { /* ignore */ }

    // 尝试方式2: 用户当前工作目录
    String userDir = System.getProperty("user.dir");
    if (userDir != null) {
        File f = new File(userDir, "plot_stimulus.py");
        if (f.exists()) return f;
    }

    // 尝试方式3: 类路径根目录（打包应用常见位置）
    try {
        String classPath = System.getProperty("java.class.path");
        if (classPath != null && classPath.contains(";")) {
            String firstEntry = classPath.split(";")[0];
            File dir = new File(firstEntry).getParentFile();
            if (dir != null) {
                File f = new File(dir, "plot_stimulus.py");
                if (f.exists()) return f;
            }
        }
    } catch (Exception e) { /* ignore */ }

    // 尝试方式4: 常见的上一级目录（build/classes 的上层）
    try {
        File cur = new File(".").getAbsoluteFile();
        File parent = cur.getParentFile();
        if (parent != null) {
            File f = new File(parent, "plot_stimulus.py");
            if (f.exists()) return f;
        }
    } catch (Exception e) { /* ignore */ }

    return null;
}

/** 查找系统中的 Python 3 解释器 */
String findPython() {
    // 检查 PATH 中的 python / python3
    for (String cmd : new String[]{"python", "python3"}) {
        try {
            Process p = new ProcessBuilder(cmd, "--version")
                .redirectErrorStream(true).start();
            int code = p.waitFor();
            if (code == 0) return cmd;
        } catch (Exception e) { /* continue */ }
    }

    // 检查常见安装路径 (Windows)
    String os = System.getProperty("os.name").toLowerCase();
    if (os.contains("win")) {
        String[] commonPaths = {
            "C:\\Python39\\python.exe",
            "C:\\Python310\\python.exe",
            "C:\\Python311\\python.exe",
            "C:\\Python312\\python.exe",
            "C:\\Python313\\python.exe",
            System.getenv("LOCALAPPDATA") + "\\Programs\\Python\\Python313\\python.exe",
            System.getenv("LOCALAPPDATA") + "\\Programs\\Python\\Python312\\python.exe",
            System.getenv("LOCALAPPDATA") + "\\Programs\\Python\\Python311\\python.exe",
            System.getenv("LOCALAPPDATA") + "\\Programs\\Python\\Python310\\python.exe",
            System.getenv("LOCALAPPDATA") + "\\Programs\\Python\\Python39\\python.exe",
        };
        for (String path : commonPaths) {
            File f = new File(path);
            if (f.exists()) return path;
        }
    }

    return null;
}

/** 跨平台打开文件（默认关联程序） */
void openFile(String path) {
    try {
        String os = System.getProperty("os.name").toLowerCase();
        if (os.contains("win")) {
            Runtime.getRuntime().exec(new String[]{"rundll32", "url.dll,FileProtocolHandler", path});
        } else if (os.contains("mac")) {
            Runtime.getRuntime().exec(new String[]{"open", path});
        } else {
            Runtime.getRuntime().exec(new String[]{"xdg-open", path});
        }
    } catch (Exception e) {
        println("openFile: " + e.getMessage());
    }
}

/** 自动保存的刺激日志目录：当前 session 子目录（stimulus_logs/<时间>/） */
String getStimulusLogDir() {
    if (stimulusSessionDir != null) return stimulusSessionDir;
    File dir = new File("stimulus_logs");
    if (!dir.exists()) {
        try { dir.mkdirs(); } catch (Exception e) {
            println("Cannot create stimulus_logs: " + e.getMessage());
        }
    }
    return "stimulus_logs" + File.separator;
}

/** 确保 JavaFX 运行时已初始化 */
static void ensureJavaFXToolkit() {
    // 方法1: 尝试 Platform.startup()
    try {
        Platform.startup(() -> {});
        return;
    } catch (IllegalStateException e) {
        // 已初始化
        return;
    } catch (Exception e) {
        println("ensureJavaFXToolkit: Platform.startup failed - " + e.getMessage());
    }

    // 方法2: 如果 Platform.startup() 不可用，使用 JFXPanel 兜底
    // （JFXPanel 的构造器一定可以启动 JavaFX 运行时）
    try {
        new JFXPanel();
        println("ensureJavaFXToolkit: JFXPanel fallback succeeded");
    } catch (Exception e) {
        println("ensureJavaFXToolkit: JFXPanel fallback also failed - " + e.getMessage());
    }
}
