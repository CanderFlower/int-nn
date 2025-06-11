mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
echo "Build complete."
echo "Running the application..."
./main
