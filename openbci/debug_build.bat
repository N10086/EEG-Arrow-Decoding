@echo off
setlocal enabledelayedexpansion

set SKETCH_DIR=E:\deskbook\OpenBCI_GUI
set PROC_DIR=C:\Program Files\Processing\app
set JAVA=%PROC_DIR%\resources\jdk\bin\java.exe

echo Building classpath...
set CP=
for /R "%PROC_DIR%" %%j in (*.jar) do set CP=!CP!;%%j
for /R "%SKETCH_DIR%\libraries" %%j in (*.jar) do set CP=!CP!;%%j
set CP=!CP!;%SKETCH_DIR%\code\LSLLink.jar

if exist "%SKETCH_DIR%\build" rmdir /S /Q "%SKETCH_DIR%\build"

echo Running Commander (debug mode - showing errors)...
"%JAVA%" -cp "!CP!" processing.mode.java.Commander --sketch="%SKETCH_DIR%" --output="%SKETCH_DIR%\build" --force --build
echo Commander exit code: %ERRORLEVEL%

if exist "%SKETCH_DIR%\build\source\OpenBCI_GUI.java" (
    echo SUCCESS: Java file generated
) else (
    echo FAILURE: No Java file found
    dir "%SKETCH_DIR%\build\" 2>nul
)
