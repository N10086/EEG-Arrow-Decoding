public class DataWriterODF {
    private PrintWriter output;
    private String fname;
    private int rowsWritten;
    private DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
    protected String fileNamePrependString = "OpenBCI-RAW-";
    protected String headerFirstLineString = "%OpenBCI Raw EXG Data";

    //variation on constructor to have custom name
    DataWriterODF(String _sessionName, String _fileName) {
        // 创建 session 子目录：stimulus_logs/<_fileName>/
        String sessionDir = "stimulus_logs" + File.separator + _fileName + File.separator;
        new File(sessionDir).mkdirs();
        stimulusSessionDir = sessionDir;   // 全局变量，供 CSV/PNG 共用
        settings.setSessionPath(sessionDir);
        fname = sessionDir + fileNamePrependString;
        fname += _fileName;
        fname += ".txt";
        output = createWriter(fname);        //open the file
        writeHeader();    //add the header
        rowsWritten = 0;    //init the counter
    }

    public void writeHeader() {
        output.println(headerFirstLineString);
        output.println("%Number of channels = " + getNumberOfChannels());
        output.println("%Sample Rate = " + getSamplingRate() + " Hz");
        output.println("%Board = " + getUnderlyingBoardClass());

        String[] colNames = getChannelNames();
        
        for (int i=0; i<colNames.length; i++) {
            output.print(colNames[i]);
            output.print(", ");
        }
        output.print("Stimulus Marker");  // 视频刺激事件标记列
        output.print(", ");
        output.print("Timestamp (Formatted)");
        output.println();
        output.flush();
    }

    public void append(double[][] data) {
        //get current date time with Date()
        for (int iSample=0; iSample<data[0].length; iSample++) {
            for (int iChan=0; iChan<data.length; iChan++) {
                output.print(data[iChan][iSample]);
                output.print(", ");
            }

            // 写入刺激事件标记
            // 0=无刺激, 1=视频播放中, 2=手动刺激标记（精确到当前 sample）
            double markerToWrite = stimulusMarkerValue;
            if (pendingMarkerValue >= 0.0) {
                // 消费挂起的标记值，仅在本 sample 生效
                markerToWrite = pendingMarkerValue;
                pendingMarkerValue = -1.0; // 立即重置，后续 sample 回退到 stimulusMarkerValue
            }
            output.print(markerToWrite);
            output.print(", ");

            int timestampChan = getTimestampChannel();
            // *1000 to convert from seconds to milliserconds
            long timestampMS = (long)(data[timestampChan][iSample] * 1000.0);

            output.print(dateFormat.format(new Date(timestampMS)));
            output.println();

            rowsWritten++;
        }
    }

    public void closeFile() {
        output.close();
    }

    public int getRowsWritten() {
        return rowsWritten;
    }

    protected int getNumberOfChannels() {
        return nchan;
    }

    protected int getSamplingRate() {
        return ((Board)currentBoard).getSampleRate();
    }

    protected String getUnderlyingBoardClass() {
        return ((Board)currentBoard).getClass().getName();
    }

    protected String[] getChannelNames() {
        return ((Board)currentBoard).getChannelNames();
    }

    protected int getTimestampChannel() {
        return ((Board)currentBoard).getTimestampChannel();
    }
    
};
