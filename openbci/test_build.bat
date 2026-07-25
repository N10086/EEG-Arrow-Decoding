@echo off
set SKETCH_DIR=E:\deskbook\OpenBCI_GUI
set PROC_DIR=C:\Program Files\Processing\app
set JAVA=%PROC_DIR%\resources\jdk\bin\java.exe

echo Building classpath...
set CP=
for /R "%PROC_DIR%" %%j in (*.jar) do set CP=!CP!;%%j
for /R "%SKETCH_DIR%\libraries" %%j in (*.jar) do set CP=!CP!;%%j

echo Running Commander...
"%JAVA%" -cp "!CP!" processing.mode.java.Commander --sketch="%SKETCH_DIR%" --output="%SKETCH_DIR%\build" --force
echo Exit code: %ERRORLEVEL%
if exist "%SKETCH_DIR%\build\source\OpenBCI_GUI.java" (
    echo SUCCESS: Java file generated!
) else (
    echo FAILURE: No Java file generated.
    dir "%SKETCH_DIR%\build\" 2>nul
)
