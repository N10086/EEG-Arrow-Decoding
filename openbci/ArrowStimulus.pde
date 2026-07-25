////////////////////////////////////////////////////////////////////////////////
//
//  ArrowStimulus.pde  - 箭头视觉刺激模块
//
//  功能：使用 JavaFX Canvas 渲染指向上下左右的箭头，按伪随机顺序依次闪烁，
//        并在每次箭头出现时向 EEG 数据流写入刺激标记。
//
//  适用于事件相关电位（ERP）运动想象 / 方向 cue 范式的诱发刺激。
//
//  编码方案（与现有标记体系一致）：
//    ↑ 上 → 标记 #1 → pendingMarkerValue = 2.0 + 1/10000
//    ↓ 下 → 标记 #2 → pendingMarkerValue = 2.0 + 2/10000
//    ← 左 → 标记 #3 → pendingMarkerValue = 2.0 + 3/10000
//    → 右 → 标记 #4 → pendingMarkerValue = 2.0 + 4/10000
//
////////////////////////////////////////////////////////////////////////////////

import java.util.Random;

import javafx.stage.Stage;
import javafx.stage.StageStyle;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.layout.StackPane;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.text.TextAlignment;
import javafx.animation.AnimationTimer;


// ====================================================================
//  ArrowStimulus  - 箭头视觉刺激主控制器
// ====================================================================
class ArrowStimulus {

    // ── 方向编码 ──
    static final int DIR_UP    = 0;
    static final int DIR_DOWN  = 1;
    static final int DIR_LEFT  = 2;
    static final int DIR_RIGHT = 3;
    static final String[] DIR_NAMES   = {"A", "B", "C", "D"};
    static final int[]    DIR_NUMBERS = {1, 2, 3, 4};  // 标记编号

    // ── 可配置参数（由 showArrowConfigDialog 设置） ──
    int trialsPerDirection  = 50;   // 每个方向的试次数
    int fixationMs          = 500;  // 注视阶段时长 (ms)
    int stimulusMs          = 200;  // 箭头显示时长 (ms)
    int interStimulusMs     = 800;  // 试次间间隔 (ms)
    boolean randomOrder     = true; // 伪随机顺序
    int flickerCount        = 1;    // 闪烁次数（1=不闪烁，连续显示）

    // ── 相位状态机 ──
    //   READY  →  FIXATION  →  STIMULUS  →  ISI  →  FIXATION → ... →  DONE
    //   等待流       注视         箭头        恢复
    enum Phase { READY, FIXATION, STIMULUS, ISI, DONE }
    private Phase phase = Phase.READY;
    private long   phaseStartNano;     // System.nanoTime() 当前相位起始
    private int    currentDirection;   // 当前试次的箭头方向
    private boolean arrowVisible = true; // 当前箭头是否可见（闪烁控制）

    // ── 试次调度 ──
    private int[] trialSequence;       // 方向编码数组
    private int   totalTrials;
    private int   currentTrialIdx;     // 当前试次索引（-1 = 未开始）

    // ── 标记系统 ──
    private int markerCount = 0;

    // ── JavaFX 组件（所有操作在 JavaFX Application Thread 上） ──
    private Stage       canvasStage;
    private Canvas      canvas;
    private AnimationTimer animTimer;

    // ── Swing 控制面板 ──
    private JFrame  controlFrame;
    private JLabel  trialCountLabel;
    private JLabel  directionLabel;
    private JLabel  phaseLabel;
    private JProgressBar progressBar;
    private JButton pauseBtn;

    // ── 运行状态 ──
    private volatile boolean isRunning    = false;
    private volatile boolean isPaused     = false;
    private volatile boolean userStopped  = false;
    private boolean streamingState = false;
    private javax.swing.Timer streamCheckTimer;

    // ── 窗口位置记忆 ──
    private int winX = -1, winY = -1, winW = 900, winH = 700;

    // ---- 前一次测试残留清理 ----
    private boolean oldStimulusWasActive = false;

    // ================================================================
    //  构造 & 启动
    // ================================================================

    ArrowStimulus() {
        String sx = System.getProperty("ARROW_WIN_X");
        String sy = System.getProperty("ARROW_WIN_Y");
        String sw = System.getProperty("ARROW_WIN_W");
        String sh = System.getProperty("ARROW_WIN_H");
        if (sx != null) try { winX = Integer.parseInt(sx); } catch (Exception e) {}
        if (sy != null) try { winY = Integer.parseInt(sy); } catch (Exception e) {}
        if (sw != null) try { winW = Integer.parseInt(sw); } catch (Exception e) {}
        if (sh != null) try { winH = Integer.parseInt(sh); } catch (Exception e) {}
    }

    /** 启动箭头刺激 */
    void start() {
        if (isRunning) return;

        // ── 清理其他正在运行的刺激模式 ──
        // 停止视频标记
        if (stimController != null && stimController.isActive()) {
            println("ArrowStimulus: 停止正在运行的 StimulusController (视频标记)");
            stimController.finishCleanup();
            oldStimulusWasActive = true;
        }
        // 停止测试回放
        if (currentTestPanel != null && !currentTestPanel.isDisposed()) {
            println("ArrowStimulus: 关闭正在运行的 TestPlaybackPanel");
            currentTestPanel.close();
            currentTestPanel = null;
        }
        // 清除旧的刺激事件
        stimulusEvents.clear();

        // ── 初始化 ──
        userStopped = false;
        isPaused    = false;
        markerCount = 0;

        // 生成伪随机试次序列
        generateTrialSequence();
        totalTrials = trialSequence.length;
        currentTrialIdx = -1;
        currentDirection = -1;

        // 设置全局标记状态
        stimulusMarkerValue = 1.0;
        stimulusModeName    = "Arrows";
        pendingMarkerValue  = -1.0;

        // 创建 Swing 控制面板
        createControlFrame();

        // 初始化 JavaFX Canvas 窗口
        ensureJavaFXToolkit();
        Platform.runLater(new Runnable() {
            public void run() {
                createCanvasStage();
            }
        });

        isRunning = true;

        // 启动数据流监控 — 仅用于标记写入判断，不控制刺激播放
        startStreamCheckTimer();

        // ★ 初始为暂停状态，等待用户点击"开始"按钮
        isPaused = true;
        streamingState = (currentBoard != null && currentBoard.isStreaming());

        outputInfo("箭头刺激已就绪，点击控制面板的 [开始] 按钮开始试次");
        println("ArrowStimulus: 已就绪, " + totalTrials + " trials, "
            + (randomOrder ? "伪随机" : "顺序") + "顺序 - 等待用户点击 [开始]");
    }

    // ================================================================
    //  伪随机试次序列生成
    // ================================================================

    private void generateTrialSequence() {
        int total = trialsPerDirection * 4;
        trialSequence = new int[total];
        int idx = 0;
        for (int d = 0; d < 4; d++) {
            for (int t = 0; t < trialsPerDirection; t++) {
                trialSequence[idx++] = d;
            }
        }

        if (!randomOrder) return;  // 顺序模式

        // 1. Fisher-Yates 洗牌
        Random rnd = new Random();
        for (int i = total - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int tmp = trialSequence[i];
            trialSequence[i] = trialSequence[j];
            trialSequence[j] = tmp;
        }

        // 2. 约束修复：禁止同一方向连续出现 ≥3 次
        //    重复扫描直到无违规
        boolean fixed;
        do {
            fixed = false;
            for (int i = 0; i < total - 2; i++) {
                if (trialSequence[i] == trialSequence[i+1]
                    && trialSequence[i] == trialSequence[i+2]) {
                    // 从后面找一个不同的位置交换
                    for (int j = i + 3; j < total; j++) {
                        if (trialSequence[j] != trialSequence[i]) {
                            int tmp = trialSequence[i+2];
                            trialSequence[i+2] = trialSequence[j];
                            trialSequence[j] = tmp;
                            fixed = true;
                            break;
                        }
                    }
                }
            }
        } while (fixed);
    }

    // ================================================================
    //  Swing 控制面板
    // ================================================================

    private void createControlFrame() {
        controlFrame = new JFrame("箭头刺激控制");
        controlFrame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        controlFrame.setResizable(false);

        JPanel panel = new JPanel(new BorderLayout(10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(18, 22, 14, 22));

        // 标题
        JLabel title = new JLabel("箭头视觉刺激", SwingConstants.CENTER);
        title.setFont(new Font("Microsoft YaHei", Font.BOLD, 16));
        panel.add(title, BorderLayout.NORTH);

        // 信息区
        JPanel infoPanel = new JPanel(new GridLayout(4, 1, 4, 4));
        infoPanel.setBorder(BorderFactory.createEmptyBorder(10, 0, 10, 0));

        // 试次
        JPanel row1 = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 2));
        row1.add(new JLabel("试次:"));
        trialCountLabel = new JLabel("0 / " + totalTrials);
        trialCountLabel.setFont(new Font("Consolas", Font.BOLD, 20));
        row1.add(trialCountLabel);
        infoPanel.add(row1);

        // 方向
        JPanel row2 = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 2));
        row2.add(new JLabel("方向:"));
        directionLabel = new JLabel("—");
        directionLabel.setFont(new Font("Microsoft YaHei", Font.BOLD, 22));
        directionLabel.setForeground(new Color(0, 80, 180));
        row2.add(directionLabel);
        infoPanel.add(row2);

        // 相位
        JPanel row3 = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 2));
        row3.add(new JLabel("相位:"));
        phaseLabel = new JLabel("就绪");
        phaseLabel.setFont(new Font("Microsoft YaHei", Font.PLAIN, 14));
        row3.add(phaseLabel);
        infoPanel.add(row3);

        // 进度条
        progressBar = new JProgressBar(0, totalTrials);
        progressBar.setStringPainted(true);
        progressBar.setFont(new Font("Consolas", Font.PLAIN, 12));
        infoPanel.add(progressBar);

        panel.add(infoPanel, BorderLayout.CENTER);

        // 按钮行
        JPanel btnRow = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 4));

        pauseBtn = new JButton("▶ 开始");
        pauseBtn.setFont(new Font("Microsoft YaHei", Font.BOLD, 13));
        pauseBtn.setFocusPainted(false);
        pauseBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                togglePause();
            }
        });
        btnRow.add(pauseBtn);

        JButton stopBtn = new JButton("■ 停止");
        stopBtn.setFont(new Font("Microsoft YaHei", Font.BOLD, 13));
        stopBtn.setBackground(new Color(200, 50, 50));
        stopBtn.setForeground(Color.WHITE);
        stopBtn.setFocusPainted(false);
        stopBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                requestStop();
            }
        });
        btnRow.add(stopBtn);

        panel.add(btnRow, BorderLayout.SOUTH);

        controlFrame.add(panel);
        controlFrame.pack();
        controlFrame.setSize(320, 260);
        controlFrame.setLocationRelativeTo(null);
        controlFrame.setVisible(true);

        controlFrame.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) { requestStop(); }
        });
    }

    private void togglePause() {
        if (!isRunning) return;

        if (currentTrialIdx < 0) {
            // ★ 初始状态：用户点击"开始"，启动试次序列
            isPaused = false;
            phase = Phase.READY;   // 确保状态机从 READY 开始
            long now = System.nanoTime();
            advanceToNextTrial(now);
            pauseBtn.setText("⏸ 暂停");
            outputInfo("箭头刺激已开始");
            println("ArrowStimulus: 用户点击[开始]，试次序列启动");
        } else {
            // 运行中：暂停/继续
            isPaused = !isPaused;
            pauseBtn.setText(isPaused ? "▶ 继续" : "⏸ 暂停");
            if (!isPaused) {
                phaseStartNano = System.nanoTime();
            }
            String msg = isPaused ? "箭头刺激已暂停" : "箭头刺激已继续";
            outputInfo(msg);
            println("ArrowStimulus: " + msg);
        }
    }

    // ================================================================
    //  JavaFX Canvas 窗口（箭头渲染）
    // ================================================================

    private void createCanvasStage() {
        if (canvasStage != null) return;

        // ★ 防止 Stage 关闭后 JavaFX 运行时退出（否则第二次无法打开）
        Platform.setImplicitExit(false);

        canvasStage = new Stage();
        canvasStage.setTitle("箭头刺激 (Arrow Stimulus)");
        canvasStage.initStyle(StageStyle.DECORATED);
        canvasStage.setAlwaysOnTop(false);

        // Canvas — 常驻，不随大小变化重建
        canvas = new Canvas(winW, winH);
        StackPane root = new StackPane(canvas);
        root.setStyle("-fx-background-color: #E8E8E8;");

        Scene scene = new Scene(root, winW, winH);
        canvasStage.setScene(scene);

        // 窗口大小变化时同步 Canvas 尺寸
        scene.widthProperty().addListener(new javafx.beans.value.ChangeListener<Number>() {
            public void changed(javafx.beans.value.ObservableValue<? extends Number> obs,
                                Number oldVal, Number newVal) {
                canvas.setWidth(newVal.doubleValue());
            }
        });
        scene.heightProperty().addListener(new javafx.beans.value.ChangeListener<Number>() {
            public void changed(javafx.beans.value.ObservableValue<? extends Number> obs,
                                Number oldVal, Number newVal) {
                canvas.setHeight(newVal.doubleValue());
            }
        });

        // 恢复窗口位置
        if (winX >= 0) canvasStage.setX(winX);
        if (winY >= 0) canvasStage.setY(winY);

        canvasStage.setOnCloseRequest(new javafx.event.EventHandler<javafx.stage.WindowEvent>() {
            public void handle(javafx.stage.WindowEvent e) {
                saveWindowPosition();
                requestStop();
            }
        });

        canvasStage.show();

        // 启动 AnimationTimer
        startAnimationTimer();

        // 显示就绪画面 — 等待用户点击 [开始]
        drawReadyState("就绪 - 点击控制面板 [开始] 按钮");
    }

    // ================================================================
    //  AnimationTimer — 帧驱动状态机
    // ================================================================

    private void startAnimationTimer() {
        if (animTimer != null) return;
        animTimer = new AnimationTimer() {
            public void handle(long nowNano) {
                if (!isRunning || isPaused || canvas == null) return;
                updatePhase(nowNano);
                renderCanvas();
            }
        };
        animTimer.start();
    }

    /** 状态机更新 */
    private void updatePhase(long nowNano) {
        if (phase == Phase.READY) {
            // 等待用户点击"开始"按钮，不自动开始试次
            return;
        }
        if (phase == Phase.DONE) return;

        long elapsedMs = (nowNano - phaseStartNano) / 1_000_000L;

        switch (phase) {
            case FIXATION:
                if (elapsedMs >= fixationMs) {
                    phase = Phase.STIMULUS;
                    phaseStartNano = nowNano;
                    // ★ 箭头出现 → 打标记
                    fireMarker(currentDirection);
                }
                break;

            case STIMULUS:
                // ── 闪烁控制 ──
                if (flickerCount <= 1) {
                    arrowVisible = true;
                } else {
                    long cycleMs = stimulusMs / flickerCount;   // 每个闪周期的时长
                    long posMs   = elapsedMs % cycleMs;         // 在当前周期内的位置
                    arrowVisible = posMs < cycleMs / 2;         // 前半周期亮，后半周期灭
                }
                if (elapsedMs >= stimulusMs) {
                    phase = Phase.ISI;
                    phaseStartNano = nowNano;
                }
                break;

            case ISI:
                if (elapsedMs >= interStimulusMs) {
                    advanceToNextTrial(nowNano);
                }
                break;

            default: break;
        }
    }

    /** 前进到下一个试次 */
    private void advanceToNextTrial(long nowNano) {
        currentTrialIdx++;
        if (currentTrialIdx >= totalTrials) {
            phase = Phase.DONE;
            onComplete();
            return;
        }
        currentDirection = trialSequence[currentTrialIdx];
        phase = Phase.FIXATION;
        phaseStartNano = nowNano;
        arrowVisible = true;    // 新试次开始时重置闪烁状态
        updateControlUI();
    }

    /** 全部试次完成 */
    private void onComplete() {
        println("ArrowStimulus: ★ 全部 " + totalTrials + " 个试次完成");
        outputSuccess("箭头刺激完成！共触发 " + markerCount + " 个标记");
        drawDoneState();
        updateControlUI();

        // 完成后自动停止（延迟 2s 以便用户看到完成提示）
        new Thread(new Runnable() {
            public void run() {
                try { Thread.sleep(2000); } catch (InterruptedException e) {}
                if (userStopped) return;
                SwingUtilities.invokeLater(new Runnable() {
                    public void run() { stop(); }
                });
            }
        }).start();
    }

    // ================================================================
    //  标记触发
    // ================================================================

    private void fireMarker(int direction) {
        if (direction < 0 || direction > 3) return;
        markerCount++;
        int markerNumber = DIR_NUMBERS[direction];

        if (streamingState) {
            pendingMarkerValue = 2.0 + markerNumber / 10000.0;
            stimulusEvents.add(new StimulusEvent(
                System.currentTimeMillis(), markerNumber,
                pendingMarkerValue, DIR_NAMES[direction]));
            println("ArrowStimulus: ★ 标记 #" + markerNumber + " (" + DIR_NAMES[direction] + ")"
                + " 试次 " + (currentTrialIdx + 1) + "/" + totalTrials);
        } else {
            println("ArrowStimulus: 标记 #" + markerNumber + " (" + DIR_NAMES[direction] + ")"
                + " — 未写入（数据流未激活）");
        }
    }

    // ================================================================
    //  Canvas 渲染
    // ================================================================

    private void renderCanvas() {
        if (canvas == null) return;
        GraphicsContext gc = canvas.getGraphicsContext2D();
        double w = canvas.getWidth();
        double h = canvas.getHeight();

        // 清空
        gc.setFill(javafx.scene.paint.Color.web("#E8E8E8"));
        gc.fillRect(0, 0, w, h);

        double cx = w / 2.0;
        double cy = h / 2.0;

        // 绘制注视十字（始终显示）
        drawFixationCross(gc, cx, cy);

        if (phase == Phase.READY) {
            // 等待用户点击"开始"
            gc.setFill(javafx.scene.paint.Color.DARKGRAY);
            gc.setFont(new javafx.scene.text.Font("Microsoft YaHei", 24));
            gc.setTextAlign(TextAlignment.CENTER);
            gc.fillText("就绪", cx, cy + 40);
            gc.setFont(new javafx.scene.text.Font("Microsoft YaHei", 14));
            gc.fillText("点击控制面板 [开始] 按钮", cx, cy + 70);
            return;
        }

        // 箭头 — 仅在 STIMULUS 相位且箭头可见时绘制
        if (phase == Phase.STIMULUS && arrowVisible && currentDirection >= 0) {
            double arrowSize = Math.min(w, h) * 0.13;
            double[] pos = getArrowPosition(cx, cy, Math.min(w, h), currentDirection);
            drawArrow(gc, pos[0], pos[1], arrowSize, currentDirection);
        }

        // 底部提示
        gc.setFill(javafx.scene.paint.Color.GRAY);
        gc.setFont(new javafx.scene.text.Font("Microsoft YaHei", 12));
        gc.setTextAlign(TextAlignment.CENTER);
        String trialInfo = "试次 " + (currentTrialIdx + 1) + " / " + totalTrials;
        gc.fillText(trialInfo, cx, h - 20);

        // 相位提示
        String phaseInfo = "";
        switch (phase) {
            case FIXATION: phaseInfo = "注视"; break;
            case STIMULUS: {
                String flickerInfo = flickerCount > 1 ? "(" + flickerCount + "闪) " : "";
                phaseInfo = flickerInfo + "箭头 → 标记 #" + DIR_NUMBERS[currentDirection];
                break;
            }
            case ISI:      phaseInfo = "恢复"; break;
            case DONE:     phaseInfo = "完成"; break;
            default: break;
        }
        gc.setFont(new javafx.scene.text.Font("Microsoft YaHei", 13));
        gc.fillText(phaseInfo, cx, h - 40);

        // 数据流指示
        gc.setFill(streamingState ? javafx.scene.paint.Color.GREEN : javafx.scene.paint.Color.RED);
        gc.fillOval(w - 28, 12, 14, 14);
        gc.setFill(streamingState ? javafx.scene.paint.Color.DARKGREEN : javafx.scene.paint.Color.DARKRED);
        gc.setFont(new javafx.scene.text.Font("Microsoft YaHei", 11));
        gc.setTextAlign(TextAlignment.RIGHT);
        gc.fillText(streamingState ? "记录中" : "未记录", w - 20, 34);
    }

    /** 绘制注视十字 */
    private void drawFixationCross(GraphicsContext gc, double cx, double cy) {
        double size = 14;
        gc.setStroke(javafx.scene.paint.Color.DARKGRAY);
        gc.setLineWidth(2.5);
        gc.strokeLine(cx - size, cy, cx + size, cy);
        gc.strokeLine(cx, cy - size, cx, cy + size);
    }

    /** 绘制箭头（朝向由 direction 指定，居中绘制） */
    private void drawArrow(GraphicsContext gc, double cx, double cy,
                           double size, int direction) {
        // 方向 → 旋转角度: UP=0°, DOWN=180°, LEFT=270°(-90°), RIGHT=90°
        double[] arrowAngles = {0, 180, 270, 90};
        double deg = arrowAngles[direction];
        double rad = Math.toRadians(deg);

        // 箭头参数（比例相对于 size）
        double shaftLen = size * 0.7;
        double headWidth  = size * 0.40;
        double headHeight = size * 0.45;

        // 箭头主体四个顶点（朝上为基准）
        // 坐标系：正 Y 向下
        double[][] base = {
            { 0 },                     // tip
            { -headWidth },            // left wing x
            { -headWidth * 0.35 },     // shaft left x
            { -headWidth * 0.35 },     // shaft left bottom x
            { 0 },                     // shaft bottom x
            { headWidth * 0.35 },      // shaft right bottom x
            { headWidth * 0.35 },      // shaft right x
            { headWidth },             // right wing x
        };
        double[][] rotated = new double[8][2];

        // 朝上的基准 Y
        double[] baseY = {
            -headHeight,               // tip y
            -headHeight * 0.15,        // left wing y
            -headHeight * 0.10,        // shaft left y
            shaftLen,                  // shaft left bottom y
            shaftLen,                  // shaft bottom y
            shaftLen,                  // shaft right bottom y
            -headHeight * 0.10,        // shaft right y
            -headHeight * 0.15,        // right wing y
        };

        // 旋转
        double cos = Math.cos(rad);
        double sin = Math.sin(rad);
        for (int i = 0; i < 8; i++) {
            rotated[i][0] = base[i][0] * cos - baseY[i] * sin + cx;
            rotated[i][1] = base[i][0] * sin + baseY[i] * cos + cy;
        }

        // 绘制填充
        double[] xPoints = new double[8];
        double[] yPoints = new double[8];
        for (int i = 0; i < 8; i++) {
            xPoints[i] = rotated[i][0];
            yPoints[i] = rotated[i][1];
        }

        gc.setFill(javafx.scene.paint.Color.BLACK);
        gc.fillPolygon(xPoints, yPoints, 8);

        // 描边
        gc.setStroke(javafx.scene.paint.Color.BLACK);
        gc.setLineWidth(1.5);
        double[] sx = new double[9];
        double[] sy = new double[9];
        for (int i = 0; i < 8; i++) {
            sx[i] = rotated[i][0];
            sy[i] = rotated[i][1];
        }
        sx[8] = rotated[0][0];
        sy[8] = rotated[0][1];
    }

    /** 根据方向返回箭头绘制位置（距离中心约 30% 屏幕短边偏移） */
    private double[] getArrowPosition(double cx, double cy, double screenMin, int direction) {
        double offset = screenMin * 0.30;
        switch (direction) {
            case DIR_UP:    return new double[]{cx,       cy - offset};
            case DIR_DOWN:  return new double[]{cx,       cy + offset};
            case DIR_LEFT:  return new double[]{cx - offset, cy};
            case DIR_RIGHT: return new double[]{cx + offset, cy};
            default:        return new double[]{cx, cy};
        }
    }

    /** 就绪画面（等待数据流） */
    /** Canvas 就绪画面（由于立即开始第一个试次，此画面几乎不可见，仅作为兜底） */
    private void drawReadyState(String message) {
        if (canvas == null) return;
        GraphicsContext gc = canvas.getGraphicsContext2D();
        double w = canvas.getWidth();
        double h = canvas.getHeight();

        gc.setFill(javafx.scene.paint.Color.web("#E8E8E8"));
        gc.fillRect(0, 0, w, h);

        drawFixationCross(gc, w/2, h/2);

        gc.setFill(javafx.scene.paint.Color.DARKGRAY);
        gc.setFont(new javafx.scene.text.Font("Microsoft YaHei", 24));
        gc.setTextAlign(TextAlignment.CENTER);
        gc.fillText(message, w/2, h/2 + 40);
    }

    /** 完成画面 */
    private void drawDoneState() {
        if (canvas == null) return;
        GraphicsContext gc = canvas.getGraphicsContext2D();
        double w = canvas.getWidth();
        double h = canvas.getHeight();

        gc.setFill(javafx.scene.paint.Color.web("#E8E8E8"));
        gc.fillRect(0, 0, w, h);

        gc.setFill(javafx.scene.paint.Color.rgb(0, 153, 0));
        gc.setFont(new javafx.scene.text.Font("Microsoft YaHei", 32));
        gc.setTextAlign(TextAlignment.CENTER);
        gc.fillText("✓ 刺激完成", w/2, h/2 - 10);

        gc.setFill(javafx.scene.paint.Color.DARKGRAY);
        gc.setFont(new javafx.scene.text.Font("Microsoft YaHei", 16));
        gc.fillText("共 " + totalTrials + " 个试次，触发 " + markerCount + " 个标记", w/2, h/2 + 30);
    }

    // ================================================================
    //  UI 更新（Swing）
    // ================================================================

    private void updateControlUI() {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                if (trialCountLabel != null) {
                    trialCountLabel.setText(
                        (phase == Phase.DONE ? totalTrials : (currentTrialIdx + 1))
                        + " / " + totalTrials);
                }
                if (directionLabel != null) {
                    if (phase == Phase.DONE) {
                        directionLabel.setText("完成");
                    } else if (currentDirection >= 0 && currentDirection < 4) {
                        directionLabel.setText(DIR_NAMES[currentDirection]);
                    }
                }
                if (phaseLabel != null) {
                    switch (phase) {
                        case READY:    phaseLabel.setText("就绪"); break;
                        case FIXATION: phaseLabel.setText("注视"); break;
                        case STIMULUS: phaseLabel.setText("箭头 - " + DIR_NAMES[currentDirection]); break;
                        case ISI:      phaseLabel.setText("恢复"); break;
                        case DONE:     phaseLabel.setText("✓ 全部完成"); break;
                    }
                }
                if (progressBar != null) {
                    progressBar.setValue(phase == Phase.DONE
                        ? totalTrials : (currentTrialIdx + 1));
                }
            }
        });
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
                    outputInfo("数据流已启动，标记将写入数据");
                    println("ArrowStimulus: 数据流已启动，标记将写入数据");
                } else if (!boardStreaming && streamingState) {
                    streamingState = false;
                    outputInfo("数据流已停止，标记不会写入数据");
                    println("ArrowStimulus: 数据流已停止，标记不会写入数据");
                }
            }
        });
        streamCheckTimer.start();
    }

    // ================================================================
    //  停止 & 清理
    // ================================================================

    void requestStop() {
        userStopped = true;
        stop();
    }

    private void stop() {
        if (!isRunning) return;
        isRunning = false;

        // 停止 AnimationTimer
        if (animTimer != null) {
            animTimer.stop();
            animTimer = null;
        }

        // 停止数据流监控
        if (streamCheckTimer != null) {
            streamCheckTimer.stop();
            streamCheckTimer = null;
        }

        // 清理标记状态
        stimulusMarkerValue = 0.0;
        pendingMarkerValue  = -1.0;
        stimulusModeName    = "No Video";

        // 关闭 JavaFX Canvas Stage
        if (canvasStage != null) {
            saveWindowPosition();
            final Stage s = canvasStage;
            canvasStage = null;
            Platform.runLater(new Runnable() {
                public void run() { s.close(); }
            });
        }

        // 关闭 Swing 控制面板
        if (controlFrame != null) {
            controlFrame.dispose();
            controlFrame = null;
        }

        // 清理全局引用
        if (currentArrowStimulus == this) {
            currentArrowStimulus = null;
        }

        int completedTrials = (currentTrialIdx >= 0) ? (currentTrialIdx + 1) : 0;
        if (userStopped && completedTrials > 0) {
            outputInfo("箭头刺激已手动停止: 已完成 " + completedTrials + "/" + totalTrials
                + " 试次, 触发 " + markerCount + " 个标记");
        }
        println("ArrowStimulus: 已停止. 试次=" + completedTrials + "/" + totalTrials
            + ", 标记=" + markerCount);
    }

    // ================================================================
    //  窗口位置保存
    // ================================================================

    void saveWindowPosition() {
        if (canvasStage == null) return;
        try {
            winX = (int) canvasStage.getX();
            winY = (int) canvasStage.getY();
            winW = (int) canvasStage.getWidth();
            winH = (int) canvasStage.getHeight();
            System.setProperty("ARROW_WIN_X", String.valueOf(winX));
            System.setProperty("ARROW_WIN_Y", String.valueOf(winY));
            System.setProperty("ARROW_WIN_W", String.valueOf(winW));
            System.setProperty("ARROW_WIN_H", String.valueOf(winH));
        } catch (Exception e) {
            println("ArrowStimulus: saveWindowPosition error - " + e.getMessage());
        }
    }
}


// ====================================================================
//  全局引用（由 StimulusLauncher 管理）
// ====================================================================
ArrowStimulus currentArrowStimulus = null;


// ====================================================================
//  配置对话框
// ====================================================================

/**
 * 显示箭头刺激参数配置对话框（在 Swing EDT 上调用）
 */
void showArrowConfigDialog() {
    // 检查是否有其他模式正在运行
    if (stimController != null && stimController.isActive()) {
        int choice = showAlwaysOnTopConfirm(
            "视频标记正在运行，是否先停止？",
            "确认", JOptionPane.YES_NO_OPTION);
        if (choice != JOptionPane.YES_OPTION) return;
        stimController.finishCleanup();
    }
    if (currentTestPanel != null && !currentTestPanel.isDisposed()) {
        int choice = showAlwaysOnTopConfirm(
            "测试回放正在运行，是否先关闭？",
            "确认", JOptionPane.YES_NO_OPTION);
        if (choice != JOptionPane.YES_OPTION) return;
        currentTestPanel.close();
        currentTestPanel = null;
    }
    if (currentArrowStimulus != null) {
        currentArrowStimulus.requestStop();
        currentArrowStimulus = null;
    }

    // 构建配置面板
    JPanel p = new JPanel(new GridBagLayout());
    GridBagConstraints c = new GridBagConstraints();
    c.insets = new Insets(5, 8, 5, 8);
    c.fill = GridBagConstraints.HORIZONTAL;
    c.gridx = 0; c.gridy = 0;

    p.add(new JLabel("每个方向试次数:"), c);
    c.gridx = 1;
    JTextField trialsField = new JTextField("50", 10);
    p.add(trialsField, c);

    c.gridx = 0; c.gridy = 1;
    p.add(new JLabel("注视时长 (ms):"), c);
    c.gridx = 1;
    JTextField fixField = new JTextField("500", 10);
    p.add(fixField, c);

    c.gridx = 0; c.gridy = 2;
    p.add(new JLabel("箭头显示时长 (ms):"), c);
    c.gridx = 1;
    JTextField stimField = new JTextField("200", 10);
    p.add(stimField, c);

    c.gridx = 0; c.gridy = 3;
    p.add(new JLabel("闪烁次数:"), c);
    c.gridx = 1;
    JTextField flickerField = new JTextField("1", 10);
    p.add(flickerField, c);

    c.gridx = 0; c.gridy = 4;
    JLabel flickerNote = new JLabel("(1=不闪烁，2=闪两次，3=闪三次)");
    flickerNote.setFont(flickerNote.getFont().deriveFont(Font.PLAIN, 10));
    flickerNote.setForeground(Color.GRAY);
    p.add(flickerNote, c);

    c.gridwidth = 1; c.gridx = 0; c.gridy = 5;
    p.add(new JLabel("试次间隔 (ms):"), c);
    c.gridx = 1;
    JTextField isiField = new JTextField("800", 10);
    p.add(isiField, c);

    c.gridx = 0; c.gridy = 6; c.gridwidth = 2;
    JCheckBox randomCB = new JCheckBox("伪随机顺序（同一方向不连续出现 3 次及以上）", true);
    p.add(randomCB, c);

    int result = JOptionPane.showConfirmDialog(null, p, "箭头刺激参数设置",
        JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);

    if (result != JOptionPane.OK_OPTION) return;

    // 创建并启动
    ArrowStimulus as = new ArrowStimulus();
    try { as.trialsPerDirection  = Math.max(1, Integer.parseInt(trialsField.getText().trim())); }
        catch (Exception ex) { as.trialsPerDirection = 50; }
    try { as.fixationMs         = Math.max(100, Integer.parseInt(fixField.getText().trim())); }
        catch (Exception ex) { as.fixationMs = 500; }
    try { as.stimulusMs          = Math.max(50, Integer.parseInt(stimField.getText().trim())); }
        catch (Exception ex) { as.stimulusMs = 200; }
    try { as.flickerCount        = Math.max(1, Integer.parseInt(flickerField.getText().trim())); }
        catch (Exception ex) { as.flickerCount = 1; }
    try { as.interStimulusMs     = Math.max(100, Integer.parseInt(isiField.getText().trim())); }
        catch (Exception ex) { as.interStimulusMs = 800; }
    as.randomOrder = randomCB.isSelected();

    currentArrowStimulus = as;
    as.start();
}
