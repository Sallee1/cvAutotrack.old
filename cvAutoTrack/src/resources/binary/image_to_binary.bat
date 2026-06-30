@REM 在单个 PowerShell 进程中处理全部图片，避免 15 次进程启动开销
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0generate_all.ps1" 
