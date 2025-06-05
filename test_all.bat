@echo off
setlocal enabledelayedexpansion

REM Header include directory
set INC_DIR=include

REM Source directory
set SRC_DIR=src

REM Test code directory
set TEST_DIR=tests

REM Compiler
set CC=gcc

REM Compiler flags
set CFLAGS=-w -I%INC_DIR% -Wall -Wextra -O2

REM Gather all source files from src\, excluding main.c
set SRC_FILES=
for %%f in (%SRC_DIR%\*.c) do (
    call :AddIfNotMain "%%f"
)

REM For each test file under tests\, compile + link + run
for %%t in (%TEST_DIR%\*.c) do (
    set TEST_NAME=%%~nt
    echo Compiling test: %%t
    %CC% %CFLAGS% -o "%TEST_DIR%\!TEST_NAME!.exe" "%%t" !SRC_FILES! -lurlmon
    if errorlevel 1 (
        echo Compile failed: !TEST_NAME!
        exit /b 1
    )
    echo Running test: !TEST_NAME!.exe
    "%TEST_DIR%\!TEST_NAME!.exe"
    echo -------------------------------
)

echo All tests done.
pause
exit /b 0

:AddIfNotMain
set "FILENAME=%~nx1"
if /I not "%FILENAME%"=="main.c" (
    set SRC_FILES=!SRC_FILES! %~1
)
exit /b
