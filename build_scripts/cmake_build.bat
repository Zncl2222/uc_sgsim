@echo off
setlocal

set "build_dir=cbuild"
set "is_execute=ON"

if /I "%~1"=="-s" (
  set "is_execute=OFF"
)

cmake -S . -B "%build_dir%" -DIS_EXECUTE=%is_execute% -G "MinGW Makefiles"
if errorlevel 1 exit /b 1

cmake --build "%build_dir%" --parallel
if errorlevel 1 exit /b 1

if /I "%is_execute%"=="ON" (
  echo.| "%build_dir%\c_example.exe"
  if errorlevel 1 exit /b 1

  "%build_dir%\test\unittest.exe"
  if errorlevel 1 exit /b 1
)
