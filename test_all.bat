@echo off
setlocal enabledelayedexpansion

REM 头文件目录
set INC_DIR=include

REM 源文件目录
set SRC_DIR=src

REM 测试代码目录
set TEST_DIR=tests

REM 编译器
set CC=gcc

REM 编译选项
set CFLAGS=-I%INC_DIR% -Wall -Wextra -O2

REM 找到所有源文件（src/*.c），生成源码文件列表（空格分隔）
set SRC_FILES=
for %%f in (%SRC_DIR%\*.c) do (
    set SRC_FILES=!SRC_FILES! %%f
)

REM 遍历每个测试文件，分别编译链接，生成独立可执行文件，运行
for %%t in (%TEST_DIR%\*.c) do (
    REM 这里直接用 %%~nt 获得测试文件名（不含路径和扩展名）
    set TEST_NAME=%%~nt

    REM 启用延迟扩展以正确获取变量
    setlocal enabledelayedexpansion

    echo 正在编译测试文件：%%t

    %CC% %CFLAGS% -o %TEST_DIR%\!TEST_NAME!.exe %%t !SRC_FILES!

    if errorlevel 1 (
        echo 编译失败：!TEST_NAME!
        endlocal
        exit /b 1
    )

    echo 运行测试程序：!TEST_NAME!.exe
    %TEST_DIR%\!TEST_NAME!.exe

    echo -----------------------------------------
    endlocal
)

echo 全部测试完成。
pause
