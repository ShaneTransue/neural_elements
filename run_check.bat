@echo off
cd /d "%~dp0"
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" check.py %*
if errorlevel 1 pause
