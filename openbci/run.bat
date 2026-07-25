@echo off
setlocal enabledelayedexpansion

set SKETCH_DIR=E:\deskbook\OpenBCI_GUI
set PROC_DIR=C:\Program Files\Processing\app
set JAVA=%PROC_DIR%\resources\jdk\bin\java.exe

REM Switch to sketch directory so Processing finds its data folder
cd /d "%SKETCH_DIR%"

REM Build classpath (must use !var! for delayed expansion in the loop)
set CP=%SKETCH_DIR%\build\classes
for %%j in (%SKETCH_DIR%\build_tools\*.jar) do set CP=!CP!;%%j
for /R %SKETCH_DIR%\libraries %%j in (*.jar) do set CP=!CP!;%%j
set CP=!CP!;%SKETCH_DIR%\code\LSLLink.jar

set NATIVE_DIR=%SKETCH_DIR%\build\natives

echo.
echo Starting OpenBCI GUI (VEP/AEP enabled) ...
echo Working directory: %SKETCH_DIR%
echo.

REM Clean old BrainFlow JNA temp files to avoid extraction conflicts
if exist "%TEMP%\jna-*" (
    echo Cleaning old BrainFlow native libraries...
    for /d %%d in (%TEMP%\jna-*) do rmdir /S /Q "%%d" 2>nul
)

"%JAVA%" -Djava.library.path="%NATIVE_DIR%" -Djna.tmpdir="%TEMP%\openbci_jna_%RANDOM%" --module-path "%SKETCH_DIR%\javafx\lib" --add-modules javafx.controls,javafx.media,javafx.swing -cp "%CP%" processing.core.PApplet OpenBCI_GUI

echo.
echo Application closed (exit code: %ERRORLEVEL%).
echo.
pause
