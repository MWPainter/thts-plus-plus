# thts-plus-plus
THTS Implementation in C++, with Python bindings (eventually).





## Installing gtest
Run the following in a bash shell starting from the root directory of this repository:
```
git submodule init
git submodule update

cd external/googletest
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=.
make
make install
```

If compiling with a compiler that isn't invoked using the standard `g++` program (e.g. using gcc instead of clang on OSX), then `Makefile` will need to be updated, and the `cmake` command above needs to be changed to: 
```
cmake .. -DCMAKE_INSTALL_PREFIX=. -DCMAKE_C_COMPILER=<your_c_compiler> -DCMAKE_CXX_COMPILER=<your_cxx_compiler>
```

Running on windows tested with `msys2` using the `Mingw64` environment, setup using the following `pacman` commands:
```
pacman -S --needed base-devel mingw-w64-x86_64-toolchain
pacman -S mingw-w64-x86_64-make
```

If using windows, then the `cmake` command needs to be changed to the following:
```
cmake -G "MSYS Makefiles" .. -DCMAKE_INSTALL_PREFIX=.
```




## TODO
1. Change use of Make to CMake
    1. Integrate gtest compilation into CMakeList.txt 
    2. Is more platform independent than Make
2. Add documentation
    1. Code design (seperate markdown), basically where to find things, and the core building blocks, diagrams?
    2. Using the library (cp some integration test), and point at bits of code that it uses
