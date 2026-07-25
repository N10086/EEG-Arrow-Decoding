////////////////////////////////////////////////////////////////////////////////
//
//  TestPlaybackPanel.pde  - 测试模式回放窗口
//
//  读取视频标记模块导出的文件夹（1 个视频 + 1 个 CSV），
//  验证并回放视频，在进度条上显示标记位置，支持定位到标记点。
//
//  设计说明：本 Panel 在首次创建后**不复位** JFXPanel / Scene / MediaView。
//  每次打开新文件夹只替换 MediaPlayer 实例，避免 JavaFX 内部状态残留。
//
////////////////////////////////////////////////////////////////////////////////

import javafx.stage.Stage;
import javafx.stage.StageStyle;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;

// ====================================================================
//  测试标记（从 CSV 解析）
// ====================================================================
class TestMarker {
    int number;
    long videoTimeMs;
    String display;

    TestMarker(int number, long videoTimeMs) {
        this.number = number;
        this.videoTimeMs = videoTimeMs;
        long s = videoTimeMs / 1000;
        this.display = String.format("%d:%02d.%03d", s / 60, s % 60, videoTimeMs % 1000);
    }
}

// ====================================================================
//  自定义进度条 — 在标记位置绘制三角形指示器
// ====================================================================
class MarkerSlider extends JSlider {
    java.util.List<TestMarker> markers;
    private static final Color MARKER_COLOR = new Color(220, 50, 50);

    MarkerSlider(int min, int max, int val) {
        super(min, max, val);
        setFocusable(false);
    }

    void setMarkers(java.util.List<TestMarker> markers) {
        this.markers = markers;
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (markers == null || markers.isEmpty()) return;

        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int w = getWidth() - 8;   // track width
        int x0 = 4;               // track start x
        int y = getHeight() / 2;  // vertical center

        g2.setColor(MARKER_COLOR);
        for (TestMarker m : markers) {
            double pct = (double) m.videoTimeMs / getMaximum();
            int mx = x0 + (int) (pct * w);
            // 画一个倒三角
            int[] xp = {mx, mx - 5, mx + 5};
            int[] yp = {y - 6, y + 4, y + 4};
            g2.fillPolygon(xp, yp, 3);
        }
        g2.dispose();
    }
}

// ====================================================================
//  TestPlaybackPanel  - 测试回放窗口
//  独立 JFrame 窗口（始终置顶），不嵌入其他窗口
// ====================================================================
class TestPlaybackPanel {
    private JFrame frame;
    private Stage videoStage;
    private MediaPlayer mediaPlayer;
    private MediaView mediaView;
    private MarkerSlider seekSlider;

    private JLabel infoLabel;
    private JPanel markerListPanel;

    private long durationMs;
    private final Object durationLock = new Object();
    private boolean playerReady = false;
    private volatile boolean seekingByUser = false;

    private String videoPath;
    private java.util.List<TestMarker> markers;

    private boolean disposed = false;
    boolean isDisposed() { return disposed; }

    // 标记当前是否有媒体正在加载/切换（防重入）
    private volatile boolean switching = false;

    // 倍速控制
    private double currentSpeed = 1.0;

    // ---- JavaFX 播放控制 ----
    private javafx.scene.control.Button fxPlayPauseBtn;
    private javafx.scene.control.Slider fxSeekSlider;
    private javafx.scene.control.Label fxTimeLabel;
    private javafx.scene.control.ComboBox<String> fxSpeedCombo;
    // ---- 常驻 Stage 组件（只创建一次） ----
    // mediaView 字段由 Processing 预处理自动生成，此处不重复声明
    // ---- 重试 & 日志 ----
    private String currentVideoPath;
    private volatile int mediaRetryCount = 0;
    private static final int MAX_MEDIA_RETRIES = 2;

    // 标签面板显示/隐藏
    private JPanel rootPanel;
    private JPanel topPanel;
    private JPanel bottomPanel;
    private JComponent markerScrollPane;   // 标记列表滚动区，单独隐藏/显示
    private JButton toggleMarkerBtn;
    private boolean markerPanelVisible = true;

    // ---- 自动标记触发（视频→EEG 同步） ----
    private boolean[] markersFired;                  // 哪些标记已触发
    private long lastReportedPos = -1;               // 上一帧视频位置 (ms)
    private volatile boolean seekingOrJumping = false; // 正在跳转中（不触发标记）
    private int markersFiredCount = 0;

    // ---- 同步日志（~10ms 精度） ----
    private PrintWriter syncLogWriter;
    private ArrayList<String> syncLogBuffer;
    private javafx.animation.Timeline syncTimeline;

    // ---- 数据流同步 ----
    private boolean streamingState = false;
    private javax.swing.Timer streamCheckTimer;

    // ================================================================
    //  构造函数 — 创建 GUI 并加载文件夹
    // ================================================================
    TestPlaybackPanel(String folderPath) throws Exception {
        createGUI();
        loadFolder(folderPath);
    }

    // ================================================================
    //  加载文件夹（可多次调用）
    // ================================================================
    void loadFolder(String folderPath) throws Exception {
        if (switching) {
            println("TestPlaybackPanel: 正在切换中，忽略重复请求");
            return;
        }
        switching = true;

        // 如果窗口之前被关闭了，重新创建 GUI
        if (frame == null) {
            println("TestPlaybackPanel: frame 为空，重新创建 GUI");
            disposed = false;
            createGUI();
        }

        try {
            File folder = new File(folderPath);
            if (!folder.isDirectory()) throw new Exception("所选路径不是文件夹");

            File[] files = folder.listFiles();
            if (files == null || files.length == 0) throw new Exception("文件夹为空");

            File videoFile = null;
            File csvFile = null;

            for (File f : files) {
                if (!f.isFile()) continue;
                String n = f.getName().toLowerCase();
                if (n.endsWith(".mp4") || n.endsWith(".avi") || n.endsWith(".mov")
                    || n.endsWith(".mkv") || n.endsWith(".wmv") || n.endsWith(".flv")) {
                    if (videoFile != null) throw new Exception("找到多个视频文件");
                    videoFile = f;
                } else if (n.endsWith(".csv")) {
                    csvFile = f;
                }
            }

            if (videoFile == null) throw new Exception("文件夹中未找到视频文件");
            if (csvFile == null) throw new Exception("文件夹中未找到标记 CSV 文件");

            final String newVideoPath = videoFile.getAbsolutePath();
            final java.util.List<TestMarker> newMarkers = parseCsv(csvFile);

            // ★ 清除前一次测试残留的 pendingMarkerValue
            //   （避免上一个视频的最后一个标记值被写入新测试的第一帧数据）
            pendingMarkerValue = -1.0;

            // 更新 UI 数据（EDT）
            this.videoPath = newVideoPath;
            this.markers = newMarkers;
            this.durationMs = 0;
            this.playerReady = false;

            String fileName = new File(videoPath).getName();
            frame.setTitle("测试回放 - " + fileName);
            infoLabel.setText(fileName + "  |  标记: " + markers.size());
            seekSlider.setValue(0);
            if (!markers.isEmpty()) {
                seekSlider.setMaximum((int) Math.max(1, markers.get(markers.size()-1).videoTimeMs + 5000));
                seekSlider.setMarkers(markers);
            } else {
                seekSlider.setMaximum(1000);
                seekSlider.setMarkers(null);
            }
            refreshMarkerList();

            // ★ 常驻 Stage + 加载 Media（常驻 Stage 只创建一次，MediaPlayer 可替换+重试）
            Platform.setImplicitExit(false);
            ensureJavaFXToolkit();
            println("[TestPlaybackPanel] >>> 提交 Platform.runLater...");
            Platform.runLater(new Runnable() {
                public void run() {
                    try {
                        // 1. 首次运行：创建常驻 Stage + Scene + MediaView + 控制栏
                        initFXStage();

                        // 2. 加载新视频（替换旧 MediaPlayer）
                        loadMedia(newVideoPath, newMarkers);

                    } catch (Exception e) {
                        println("[TestPlaybackPanel] runLater 异常: " + e.getMessage());
                        e.printStackTrace();
                        SwingUtilities.invokeLater(new Runnable() {
                            public void run() {
                                outputError("测试回放：无法播放视频 (" + e.getMessage() + ")");
                            }
                        });
                    } finally {
                        switching = false;
                    }
                }
            });
            println("[TestPlaybackPanel] <<< Platform.runLater 已提交");

            // ★ Platform.runLater 提交后再显示窗口，此时 JavaFX Stage 已创建
            boolean firstShow = !frame.isVisible();
            if (firstShow) {
                frame.setVisible(true);
            }
            if (streamCheckTimer == null) {
                startStreamCheckTimer();
            }
            if (!frame.isVisible()) frame.setVisible(true);
            frame.toFront();

        } catch (Exception e) {
            switching = false;
            throw e;
        }
    }

    // ================================================================
    //  常驻 JavaFX Stage（创建一次，不随视频切换重建）
    // ================================================================
    private void initFXStage() {
        if (videoStage != null) {
            println("[TestPlaybackPanel] Stage 已存在，跳过创建");
            return;
        }

        String fileName = new File(videoPath).getName();
        println("[TestPlaybackPanel] --- 创建常驻 Stage ---");

        videoStage = new Stage();
        videoStage.setTitle("测试回放 - " + fileName);
        videoStage.initStyle(StageStyle.DECORATED);
        videoStage.setAlwaysOnTop(false);

        // ★ 常驻 MediaView
        mediaView = new MediaView();
        mediaView.setPreserveRatio(true);
        mediaPlayer = null;

        // ---- FX controls ----
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

        // ---- Layout ----
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

        videoStage.setOnCloseRequest(new javafx.event.EventHandler<javafx.stage.WindowEvent>() {
            public void handle(javafx.stage.WindowEvent e) {
                println("[TestPlaybackPanel] Stage 关闭请求 → 暂停 + 隐藏");
                if (mediaPlayer != null) mediaPlayer.pause();
                videoStage.hide();
            }
        });

        if (frame != null && frame.isVisible()) {
            java.awt.Point loc = frame.getLocationOnScreen();
            videoStage.setX(loc.x + frame.getWidth() + 10);
            videoStage.setY(loc.y);
        }

        // ★ Stage 不立即 show()，等 onReady 后再显示
        println("[TestPlaybackPanel] ✓ 常驻 Stage 创建完毕（未显示，等待 onReady）");
    }

    // ================================================================
    //  加载视频（入口：切换文件夹 / 首次加载）
    // ================================================================
    private void loadMedia(String path, java.util.List<TestMarker> newMarkers) {
        println("[TestPlaybackPanel] --- loadMedia: " + path + " ---");

        currentVideoPath = path;
        mediaRetryCount = 0;

        // 更新标记状态
        this.markers = newMarkers;
        markersFired = new boolean[markers != null ? markers.size() : 0];
        markersFiredCount = 0;
        seekingOrJumping = false;
        lastReportedPos = -1;

        doLoadMedia();
    }

    // ================================================================
    //  执行 Media 加载 + 错误重试
    // ================================================================
    private void doLoadMedia() {
        println("[TestPlaybackPanel]    doLoadMedia [尝试 #" + (mediaRetryCount + 1)
            + "/" + (MAX_MEDIA_RETRIES + 1) + "]");

        // 1. 释放旧 MediaPlayer
        if (mediaPlayer != null) {
            try {
                println("[TestPlaybackPanel]    释放旧 MediaPlayer...");
                mediaPlayer.stop();
                mediaPlayer.dispose();
            } catch (Exception e) {
                println("[TestPlaybackPanel]    释放旧 MediaPlayer 异常: " + e.getMessage());
            }
            mediaPlayer = null;
        }

        // 2. 重置状态
        playerReady = false;
        synchronized (durationLock) {
            durationMs = 0;
        }

        try {
            // 3. 创建 Media
            String uri = new File(currentVideoPath).toURI().toString();
            println("[TestPlaybackPanel]    URI: " + uri);
            Media media = new Media(uri);

            // ★ Media 级别错误
            media.setOnError(new Runnable() {
                public void run() {
                    String errMsg = (media.getError() != null) ? media.getError().getMessage() : "未知";
                    println("[TestPlaybackPanel]    ✗ [Media.onError] " + errMsg);
                }
            });

            // 4. 创建新 MediaPlayer
            MediaPlayer newPlayer = new MediaPlayer(media);
            newPlayer.setRate(currentSpeed);
            println("[TestPlaybackPanel]    MediaPlayer 已创建");

            // ---- sync log timeline ----
            startSyncTimeline();

            // ★ 当前时间 → 进度条 + 标记触发 + Swing 同步
            newPlayer.currentTimeProperty().addListener(new javafx.beans.value.ChangeListener<Duration>() {
                public void changed(javafx.beans.value.ObservableValue<? extends Duration> obs, Duration old, Duration now) {
                    if (!seekingByUser && playerReady) {
                        final long cur = (long) now.toMillis();
                        checkAndFireMarkers(cur);
                        synchronized (durationLock) {
                            if (durationMs > 0) {
                                double pct = (double) cur / durationMs;
                                fxSeekSlider.setValue(pct);
                                fxTimeLabel.setText(formatTime(cur) + " / " + formatTime(durationMs));
                                // 同步 Swing 标记进度条
                                final long fCur = cur;
                                final long fTotal = durationMs;
                                SwingUtilities.invokeLater(new Runnable() {
                                    public void run() {
                                        seekSlider.setValue((int) Math.min(fCur, fTotal));
                                    }
                                });
                            }
                        }
                    }
                }
            });

            // ★ 状态监听（播放/暂停 按钮 + 日志）
            newPlayer.statusProperty().addListener(new javafx.beans.value.ChangeListener<MediaPlayer.Status>() {
                public void changed(javafx.beans.value.ObservableValue<? extends MediaPlayer.Status> obs, MediaPlayer.Status old, MediaPlayer.Status stat) {
                    fxPlayPauseBtn.setText(stat == MediaPlayer.Status.PLAYING ? "⏸ 暂停" : "▶ 播放");
                    if (stat == MediaPlayer.Status.PLAYING) {
                        println("[TestPlaybackPanel] ▶ [status=PLAYING]");
                    } else if (stat == MediaPlayer.Status.PAUSED) {
                        println("[TestPlaybackPanel] ⏸ [status=PAUSED]");
                    } else if (stat == MediaPlayer.Status.STOPPED) {
                        println("[TestPlaybackPanel] ⏹ [status=STOPPED]");
                    } else if (stat == MediaPlayer.Status.READY) {
                        println("[TestPlaybackPanel] ✓ [status=READY]");
                    }
                }
            });

            // 5. ★ onReady
            newPlayer.setOnReady(new Runnable() {
                public void run() {
                    synchronized (durationLock) {
                        durationMs = (long) newPlayer.getTotalDuration().toMillis();
                    }
                    playerReady = true;
                    mediaRetryCount = 0;
                    println("[TestPlaybackPanel] ✓ [onReady] 视频已就绪, 时长=" + durationMs + "ms");

                    // 更新 Swing 进度条范围
                    if (durationMs > 0) {
                        final long totalMs = durationMs;
                        SwingUtilities.invokeLater(new Runnable() {
                            public void run() {
                                seekSlider.setMaximum((int) totalMs);
                                seekSlider.setEnabled(true);
                                println("[TestPlaybackPanel] → Swing seekSlider 已更新: max=" + totalMs);
                            }
                        });
                    }

                    // 首次或重新显示 Stage
                    if (videoStage != null && !videoStage.isShowing()) {
                        println("[TestPlaybackPanel] → 首次显示 Stage (onReady 后)");
                        videoStage.show();
                    }
                }
            });

            // 6. ★★★ onError + 自动重试 ★★★
            newPlayer.setOnError(new Runnable() {
                public void run() {
                    String errMsg = "未知";
                    try {
                        Throwable err = newPlayer.getError();
                        if (err != null) {
                            errMsg = err.getMessage();
                            if (errMsg == null) errMsg = err.getClass().getName();
                            println("[TestPlaybackPanel] 错误详情: " + errMsg);
                        }
                    } catch (Exception ex) {
                        errMsg = ex.getMessage();
                        if (errMsg == null) errMsg = "未知异常";
                    }
                    println("[TestPlaybackPanel] ⚠ [onError] " + errMsg);

                    if (mediaRetryCount < MAX_MEDIA_RETRIES) {
                        mediaRetryCount++;
                        println("[TestPlaybackPanel] ⚠ 500ms 后自动重试第 " + mediaRetryCount + " 次...");
                        javafx.animation.PauseTransition pt = new javafx.animation.PauseTransition(Duration.millis(500));
                        pt.setOnFinished(new javafx.event.EventHandler<javafx.event.ActionEvent>() {
                            public void handle(javafx.event.ActionEvent e) {
                                doLoadMedia();
                            }
                        });
                        pt.play();
                    } else {
                        println("[TestPlaybackPanel] ✗ 已达最大重试次数 (" + MAX_MEDIA_RETRIES + ")，放弃");
                        outputError("测试回放失败（已重试 " + MAX_MEDIA_RETRIES + " 次）: " + errMsg);
                        if (videoStage != null && !videoStage.isShowing()) {
                            println("[TestPlaybackPanel] → 兜底显示 Stage");
                            videoStage.show();
                        }
                    }
                }
            });

            // 7. 绑定到常驻 MediaView
            mediaView.setMediaPlayer(newPlayer);
            mediaPlayer = newPlayer;

            println("[TestPlaybackPanel]    MediaPlayer 绑定完毕，等待 onReady...");

        } catch (Exception e) {
            println("[TestPlaybackPanel]    ✗ doLoadMedia 异常: " + e.getMessage());
            e.printStackTrace();
            if (mediaRetryCount < MAX_MEDIA_RETRIES) {
                mediaRetryCount++;
                println("[TestPlaybackPanel] ⚠ 500ms 后异常恢复重试第 " + mediaRetryCount + " 次...");
                javafx.animation.PauseTransition pt = new javafx.animation.PauseTransition(Duration.millis(500));
                pt.setOnFinished(new javafx.event.EventHandler<javafx.event.ActionEvent>() {
                    public void handle(javafx.event.ActionEvent e) {
                        doLoadMedia();
                    }
                });
                pt.play();
            } else {
                outputError("测试回放加载异常（已重试 " + MAX_MEDIA_RETRIES + " 次）: " + e.getMessage());
            }
        }
    }

    // ---- CSV 解析 ----
    private java.util.List<TestMarker> parseCsv(File csvFile) throws Exception {
        java.util.List<TestMarker> list = new java.util.ArrayList<TestMarker>();
        java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(csvFile));
        String line;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] parts = line.split(",");
            if (parts.length >= 2) {
                try {
                    int num = Integer.parseInt(parts[0].trim());
                    long timeMs = Long.parseLong(parts[1].trim());
                    list.add(new TestMarker(num, timeMs));
                } catch (Exception e) { /* 跳过格式错误的行 */ }
            }
        }
        br.close();
        println("TestPlaybackPanel: Parsed " + list.size() + " markers from " + csvFile.getName());
        return list;
    }

    // ---- 创建 GUI（只执行一次） ----
    private void createGUI() {
        frame = new JFrame("测试回放");
        frame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        frame.setSize(640, 520);
        frame.setMinimumSize(new Dimension(440, 360));
        frame.setLocationRelativeTo(null);
        // ★ 窗口始终置顶，避免藏在主窗口后面
        frame.setAlwaysOnTop(true);

        buildUI();
        frame.add(rootPanel);

        frame.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) { close(); }
        });

        // ★ 窗口在 loadFolder 验证通过后才显示
    }

    // ---- 构建 UI 面板 ----
    private void buildUI() {
        rootPanel = new JPanel(new BorderLayout(6, 6));
        rootPanel.setBorder(BorderFactory.createEmptyBorder(8, 8, 5, 8));
        JPanel root = rootPanel;

        // ── 顶部信息 ──
        topPanel = new JPanel(new BorderLayout(6, 3));
        infoLabel = new JLabel("请选择标记文件夹");
        infoLabel.setFont(new Font("Microsoft YaHei", Font.PLAIN, 12));
        topPanel.add(infoLabel, BorderLayout.WEST);
        root.add(topPanel, BorderLayout.NORTH);

        // ---- 标记位置指示器 ----
        seekSlider = new MarkerSlider(0, 1000, 0);
        seekSlider.setEnabled(false);
        seekSlider.addMouseListener(new MouseAdapter() {
            public void mousePressed(java.awt.event.MouseEvent e) { seekingByUser = true; }
            public void mouseReleased(java.awt.event.MouseEvent e) {
                seekingByUser = false;
                seekToPosition(Math.max(0, seekSlider.getValue()));
            }
        });
        root.add(seekSlider, BorderLayout.CENTER);

        // ---- 底部：标记列表 + 按钮 ----
        bottomPanel = new JPanel(new BorderLayout(5, 5));
        markerListPanel = new JPanel();
        markerListPanel.setLayout(new BoxLayout(markerListPanel, BoxLayout.Y_AXIS));
        markerScrollPane = new JScrollPane(markerListPanel);
        markerScrollPane.setBorder(BorderFactory.createTitledBorder("标记列表（点击定位）"));
        markerScrollPane.setPreferredSize(new Dimension(360, 130));
        bottomPanel.add(markerScrollPane, BorderLayout.CENTER);

        JPanel btnRow = new JPanel(new FlowLayout(FlowLayout.CENTER, 6, 3));
        JButton selectBtn = new JButton("选择其他文件夹");
        selectBtn.setFont(new Font("Microsoft YaHei", Font.PLAIN, 12));
        selectBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                // 使用始终顶层的原生文件选择器
                new Thread(new Runnable() {
                    public void run() {
                        File folder = showFileChooserBlocking("选择刺激标记文件夹", true, null);
                        if (folder != null) {
                            final File f = folder;
                            SwingUtilities.invokeLater(new Runnable() {
                                public void run() {
                                    onReloadFolderSelected(f);
                                }
                            });
                        }
                    }
                }).start();
            }
        });
        btnRow.add(selectBtn);

        toggleMarkerBtn = new JButton("▼ 隐藏标记");
        toggleMarkerBtn.setFont(new Font("Microsoft YaHei", Font.PLAIN, 12));
        toggleMarkerBtn.setFocusPainted(false);
        toggleMarkerBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) { toggleMarkerPanel(); }
        });
        btnRow.add(toggleMarkerBtn);

        JButton closeBtn = new JButton("关闭");
        closeBtn.setFont(new Font("Microsoft YaHei", Font.PLAIN, 12));
        closeBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) { close(); }
        });
        btnRow.add(closeBtn);
        bottomPanel.add(btnRow, BorderLayout.SOUTH);

        root.add(bottomPanel, BorderLayout.SOUTH);

        markerListPanel.add(new JLabel("  暂无标记"));
    }

    // ---- 标记列表 ----
    private void refreshMarkerList() {
        markerListPanel.removeAll();
        if (markers == null || markers.isEmpty()) {
            markerListPanel.add(new JLabel("  暂无标记"));
        } else {
            for (int i = markers.size() - 1; i >= 0; i--) {
                final TestMarker m = markers.get(i);
                JPanel row = new JPanel(new BorderLayout(6, 2));
                row.setBorder(BorderFactory.createEmptyBorder(2, 4, 2, 4));

                JLabel lab = new JLabel(" #" + m.number + "  " + m.display);
                lab.setFont(new Font("Consolas", Font.PLAIN, 13));
                row.add(lab, BorderLayout.CENTER);

                JButton jumpBtn = new JButton("定位");
                jumpBtn.setFont(new Font("Microsoft YaHei", Font.PLAIN, 11));
                jumpBtn.setFocusPainted(false);
                jumpBtn.addActionListener(new ActionListener() {
                    public void actionPerformed(ActionEvent e) {
                        seekToPosition(m.videoTimeMs);
                    }
                });
                row.add(jumpBtn, BorderLayout.EAST);

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

    // ================================================================
    //  播放控制
    // ================================================================
    private void seekToPosition(long timeMs) {
        if (mediaPlayer == null || !playerReady) return;
        seekingByUser = true;
        seekingOrJumping = true;
        seekSlider.setValue((int) Math.min(timeMs, durationMs > 0 ? durationMs : seekSlider.getMaximum()));
        seekingByUser = false;
        Platform.runLater(new Runnable() {
            public void run() {
                mediaPlayer.seek(Duration.millis(timeMs));
            }
        });
    }

    private String formatTime(long ms) {
        long s = ms / 1000;
        return String.format("%d:%02d.%03d", s / 60, s % 60, ms % 1000);
    }

    // ================================================================
    //  显示/隐藏底部标签面板
    // ================================================================
    private void toggleMarkerPanel() {
        markerPanelVisible = !markerPanelVisible;
        if (markerScrollPane != null) {
            markerScrollPane.setVisible(markerPanelVisible);
            bottomPanel.revalidate();
            bottomPanel.repaint();
        }
        toggleMarkerBtn.setText(markerPanelVisible ? "▼ 隐藏标记" : "▲ 显示标记");
    }

    // ================================================================
    //  清理
    // ================================================================
    void close() {
        if (disposed) return;
        println("[TestPlaybackPanel] === close() ===");
        disposed = true;

        // 停止同步日志
        stopSyncLog();
        stopSyncTimeline();
        if (streamCheckTimer != null) {
            streamCheckTimer.stop();
            streamCheckTimer = null;
        }

        // 先同步停掉 JavaFX 侧的 MediaPlayer，再销毁 Swing 容器
        if (mediaPlayer != null) {
            println("[TestPlaybackPanel] 停止 + 释放 MediaPlayer...");
            final MediaPlayer mp = mediaPlayer;
            mediaPlayer = null;
            final java.util.concurrent.CountDownLatch disposeLatch =
                new java.util.concurrent.CountDownLatch(1);
            Platform.runLater(new Runnable() {
                public void run() {
                    try {
                        if (videoStage != null) {
                            println("[TestPlaybackPanel] 关闭 Stage...");
                            videoStage.close();
                            videoStage = null;
                        }
                        println("[TestPlaybackPanel] dispose MediaPlayer...");
                        mp.stop();
                        mp.dispose();
                        println("[TestPlaybackPanel] ✓ MediaPlayer 已释放");
                    } catch (Exception e) {
                        println("[TestPlaybackPanel] [close] dispose 异常 - " + e.getMessage());
                    } finally {
                        disposeLatch.countDown();
                    }
                }
            });
            try {
                disposeLatch.await(3, java.util.concurrent.TimeUnit.SECONDS);
                println("[TestPlaybackPanel] ✓ dispose 完成");
            } catch (InterruptedException ie) {
                println("[TestPlaybackPanel] [close] await 被中断");
            }
        } else {
            // 即使没有 MediaPlayer，清理 Stage
            if (videoStage != null) {
                println("[TestPlaybackPanel] 关闭 Stage (no MediaPlayer)...");
                final Stage vs = videoStage;
                videoStage = null;
                Platform.runLater(new Runnable() {
                    public void run() { vs.close(); }
                });
            }
        }

        if (frame != null) {
            println("[TestPlaybackPanel] 关闭 Swing Frame...");
            frame.dispose();
            frame = null;
        }
        println("[TestPlaybackPanel] === close() 完成 ===");
    }

    // ================================================================
    //  数据流状态监控
    // ================================================================
    private void startStreamCheckTimer() {
        if (streamCheckTimer != null) return; // 已有 timer，防止重复
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
                                // 补注入
                                retroactivelyInjectMarkers();
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
    //  标记自动触发（视频自然播放到标记点时调用）
    // ================================================================
    void checkAndFireMarkers(long currentPos) {
        // seekingOrJumping 已检测跳转/拖动进度条场景，此时不触发标记
        if (markers == null || markersFired == null || seekingOrJumping) return;

        // 检查从 lastReportedPos 到 currentPos 之间的所有标记。
        // 【注意】JavaFX MediaPlayer 的 currentTimeProperty 在系统负载高时
        // 触发间隔可能超过 100ms，因此不使用时间间隔过滤（maxNormalDelta）。
        // seekingOrJumping 已正确处理跳转防误触，delta 过滤是多余的。
        if (lastReportedPos >= 0) {
            for (int i = 0; i < markers.size(); i++) {
                if (i >= markersFired.length) break;
                TestMarker m = markers.get(i);
                if (!markersFired[i] && lastReportedPos < m.videoTimeMs && currentPos >= m.videoTimeMs) {
                    if (streamingState) {
                        markersFired[i] = true;
                        markersFiredCount++;
                        pendingMarkerValue = 2.0 + m.number / 10000.0;
                        writeSyncLogMarker(m.number, currentPos);
                        stimulusEvents.add(new StimulusEvent(System.currentTimeMillis(), m.number, pendingMarkerValue));
                    }
                    println("TestPlaybackPanel: 自动触发标记 #" + m.number + " @ " + m.display);
                }
            }
        }

        lastReportedPos = currentPos;
    }

    void retroactivelyInjectMarkers() {
        if (markers == null || markersFired == null) return;
        long curPos = (mediaPlayer != null)
            ? (long) mediaPlayer.getCurrentTime().toMillis()
            : 0;
        boolean anyInjected = false;
        for (int i = 0; i < markers.size(); i++) {
            if (i >= markersFired.length) break;
            TestMarker m = markers.get(i);
            if (!markersFired[i] && curPos >= m.videoTimeMs) {
                markersFired[i] = true;
                markersFiredCount++;
                pendingMarkerValue = 2.0 + m.number / 10000.0;
                writeSyncLogMarker(m.number, curPos);
                stimulusEvents.add(new StimulusEvent(System.currentTimeMillis(), m.number, pendingMarkerValue));
                println("TestPlaybackPanel: 补注入标记 #" + m.number + " @ " + m.display);
                anyInjected = true;
            }
        }
        if (anyInjected) {
            println("TestPlaybackPanel: 补注入完成，共 " + markersFiredCount + " 个标记");
        }
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

                if (seekingOrJumping) {
                    long delta = Math.abs(cur - lastReportedPos);
                    if (delta < 100 && lastReportedPos >= 0) {
                        seekingOrJumping = false;
                    }
                    lastReportedPos = cur;
                }

                if (syncLogWriter != null) {
                    boolean playing = mediaPlayer.getStatus() == MediaPlayer.Status.PLAYING;
                    syncLogBuffer.add(System.currentTimeMillis() + "," + cur + "," + (playing ? "1" : "0") + ",");
                    if (syncLogBuffer.size() >= 100) flushSyncLog();
                }
            }
        }));
        syncTimeline.play();
    }

    private void writeSyncLogMarker(int markerNumber, long videoMs) {
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
