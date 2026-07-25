////////////////////////////////////////////////////////////////////////////////
//
//  StimulusLauncher.pde  - 刺激实验控制台主界面
//
//  提供四个入口：
//    1. 视频标记 — 导入视频并实时打标记（原有功能）
//    2. 测试     — 读取已保存的标记文件夹，回放验证
//    3. 箭头刺激 — 在屏幕中闪烁上下左右箭头，同步记录 EEG 标记
//    4. 数据分析 — 读取 ODF .txt 生成刺激标注波形图
//
//  注意：点击"测试"、"箭头刺激"或"数据分析"后本窗口会关闭，
//        各自的功能在新窗口中运行（均 setAlwaysOnTop）。
////////////////////////////////////////////////////////////////////////////////

class StimulusLauncher {
    private JFrame frame;
    private boolean isOpen = false;

    void open() {
        if (isOpen && frame != null) {
            frame.toFront();
            return;
        }
        createAndShow();
    }

    void close() {
        if (frame != null) {
            frame.dispose();
            frame = null;
        }
        isOpen = false;
    }

    boolean isOpen() { return isOpen; }

    // ---- GUI ----
    private void createAndShow() {
        frame = new JFrame("刺激实验控制台");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setSize(380, 390);
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        frame.setAlwaysOnTop(true);

        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setBorder(BorderFactory.createEmptyBorder(20, 28, 16, 28));

        // 标题
        JLabel title = new JLabel("刺激实验控制台", SwingConstants.CENTER);
        title.setFont(new Font("Microsoft YaHei", Font.BOLD, 17));
        title.setAlignmentX(Component.CENTER_ALIGNMENT);
        panel.add(title);
        panel.add(Box.createVerticalStrut(18));

        // ── 视频标记按钮 ──
        JButton markBtn = createLauncherButton("📹  视频标记",
            "导入视频并实时打标记 / 暂停 / 导出");
        markBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                close();
                if (stimController != null) stimController.importVideo();
            }
        });
        panel.add(markBtn);
        panel.add(Box.createVerticalStrut(10));

        // ── 测试按钮 ──
        JButton testBtn = createLauncherButton("📊  测试",
            "读取已保存的标记文件夹，回放视频并显示标记");
        testBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                close();
                // 使用始终顶层的原生文件选择器
                new Thread(new Runnable() {
                    public void run() {
                        File folder = showFileChooserBlocking("选择刺激标记文件夹", true, null);
                        if (folder != null) {
                            final File f = folder;
                            SwingUtilities.invokeLater(new Runnable() {
                                public void run() {
                                    onTestFolderSelected(f);
                                }
                            });
                        }
                    }
                }).start();
            }
        });
        panel.add(testBtn);
        panel.add(Box.createVerticalStrut(10));

        // ── 箭头刺激按钮 ──
        JButton arrowBtn = createLauncherButton("➡️  箭头刺激",
            "在屏幕中按伪随机顺序闪烁上下左右箭头，同步记录 EEG 标记");
        arrowBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                close();
                new Thread(new Runnable() {
                    public void run() {
                        SwingUtilities.invokeLater(new Runnable() {
                            public void run() {
                                showArrowConfigDialog();
                            }
                        });
                    }
                }).start();
            }
        });
        panel.add(arrowBtn);
        panel.add(Box.createVerticalStrut(10));

        // ── 数据分析按钮 ──
        JButton analysisBtn = createLauncherButton("📈  数据分析",
            "读取已保存的 TXT 录制文件，生成刺激标注波形图");
        analysisBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                close();
                String recPath = (directoryManager != null)
                    ? directoryManager.getRecordingsPath() : System.getProperty("user.dir");
                final String rp = recPath;
                // 使用始终顶层的原生文件选择器
                new Thread(new Runnable() {
                    public void run() {
                        File txtFile = showFileChooserBlocking("选择 ODF 数据文件 (.txt)", false, new File(rp));
                        if (txtFile != null) {
                            final File f = txtFile;
                            SwingUtilities.invokeLater(new Runnable() {
                                public void run() {
                                    onAnalysisFileSelected(f);
                                }
                            });
                        }
                    }
                }).start();
            }
        });
        panel.add(analysisBtn);

        panel.add(Box.createVerticalGlue());

        // 提示文字
        JLabel tip = new JLabel("视频标记 → 导出文件夹 → 测试回放", SwingConstants.CENTER);
        tip.setFont(new Font("Microsoft YaHei", Font.PLAIN, 11));
        tip.setForeground(Color.GRAY);
        tip.setAlignmentX(Component.CENTER_ALIGNMENT);
        panel.add(tip);

        frame.add(panel);
        frame.setVisible(true);
        isOpen = true;

        frame.addWindowListener(new WindowAdapter() {
            public void windowClosed(WindowEvent e) { isOpen = false; }
        });
    }

    private JButton createLauncherButton(String text, String tooltip) {
        JButton btn = new JButton(text);
        btn.setFont(new Font("Microsoft YaHei", Font.BOLD, 15));
        btn.setPreferredSize(new Dimension(300, 52));
        btn.setMaximumSize(new Dimension(300, 52));
        btn.setMinimumSize(new Dimension(300, 40));
        btn.setAlignmentX(Component.CENTER_ALIGNMENT);
        btn.setFocusPainted(false);
        btn.setToolTipText(tooltip);
        return btn;
    }
}
