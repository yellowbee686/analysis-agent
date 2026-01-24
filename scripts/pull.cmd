@echo off
setlocal

where bash >nul 2>nul
if errorlevel 1 (
  echo ERROR: bash not found. Please install Git Bash or WSL and ensure bash is in PATH.
  exit /b 1
)

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:\=/%"

pushd "%~dp0.." >nul
bash "%SCRIPT_DIR%pull.sh"
set "EXITCODE=%ERRORLEVEL%"
popd >nul

exit /b %EXITCODE%
