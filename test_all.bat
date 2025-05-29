@echo off
setlocal enabledelayedexpansion

REM 头文件目录
set INC_DIR=include

REM 源文件目录
set SRC_DIR=src

REM 测试代码目录
set TEST_DIR=tests

REM 编译器（这里用gcc，如果用MSVC可以改）
set CC=gcc

REM 输出可执行文件名
set EXE_NAME=test_intnn_mat.exe

REM 收集所有源文件
set SRC_FILES=

for %%f in (%SRC_DIR%\*.c) do (
    set SRC_FILES=!SRC_FILES! %%f
)

REM 编译并链接
echo 编译并链接测试程序...
%CC% -I%INC_DIR% -o %TEST_DIR%\%EXE_NAME% %SRC_FILES% %TEST_DIR%\test_intnn_mat.c -Wall -Wextra -O2

if errorlevel 1 (
    echo 编译失败！
    exit /b 1
)

echo 运行测试程序...
%TEST_DIR%\%EXE_NAME%

pause
