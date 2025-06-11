@echo off
if exist build (
    echo Deleting existing build folder...
    rmdir /s /q build
)
mkdir build
cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
mingw32-make
echo Build complete.
echo Running the application...
main.exe
cd ..