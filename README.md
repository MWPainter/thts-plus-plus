# thts-plus-plus
THTS Implementation in C++, with Python bindings (eventually). By default, running `make` will just compile unit tests to an executable called `thts-test`, if you run this some tests will fail but they should all contain `todo` in their names, as they are mostly placeholder tests.



## Code Overview

All subdirectories in the `include/` folder contain `README.md` files, which give an overview of the code from that 
directory. Currently a work in progress, so lots of barebone notes rather than documentation (tends to get more bare 
the deeper into directories you get). 

Also code includes pretty heavy commenting, so hopefully most functions have descriptive enough headers to know how 
to use and interact with the classes. Additionally, plenty of functions have comments in the implementation files 



## Installing gtest
Run the following in a bash shell starting from the root directory of this repository:
```
git submodule init
git submodule update
```
```
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

Running on windows tested with `msys2` using the `Mingw64` environment, setup using the following `pacman` commands. If using windows, then the `cmake` command above needs to be changed to the following as well:
```
pacman -S --needed base-devel mingw-w64-x86_64-toolchain
pacman -S mingw-w64-x86_64-make
```
```
cmake -G "MSYS Makefiles" .. -DCMAKE_INSTALL_PREFIX=.
```



## Changelog

This is a space to say what's included and updated in any versioning happening. I'll create the v0.1 branch when the 
repository is in a state where I feel comfortable handing it over to other people. v1.0 are things that I would like to 
have in some sort of 'full release', but who know's if we'll ever get to that point :)

### v0.1

The initial implementation of this library, including the following:
- Core datatypes, including `ThtsEnv` to encapsulate the environment interface for being able to use these THTS routines
- A generalised multi-threaded implementation of THTS (for example includes changes allowing for POMDP algorithms)
- Unit tests on most core functionality
- A basic monte-carlo evaluation class
- Implementations of the following algorithms (with some integration tests, but no unit tests):
    - UCT
    - Polynomial UCT
    - MENTS
    - RENTS
    - TENTS
- Including novel algorithms implemented in this repository:
    - DENTS
    - DB-MENTS
    - EST
- Additionally, `ThtsManager` objects provide a lot of parameters in these algorithms to be customised
    - Including customisations that aren't necessarily strictly defined by the original algorithms, such as having an epsilon exploration component to UCT, and being able to decay the temperature parameter used in MENTS. None of these additional components are used by default.

### Todos

Moved all todos to github issues, most of them are marked with v0.1 and v1.0. If you are using this repo and see 
something that could be improved, or even just better explained/documented please add your own issues (or message me, 
but I'm pretty bad at responding), I am very keen for this library to be easy to use for other people.



# Go Stuff

To build, do the following steps:
1. Run cmake in external/KataGo/cpp: cmake . -D USE_BACKEND=<your_backend_eg_CUDA> -D CMAKE_CXX_COMPILER=<your_cxx_compiler> -D CMAKE_C_COMPILER=<your_c_compiler>
2. optional on cpu: -DUSE_AVX2=1 -DCMAKE_CXX_FLAGS='-march=native'
2. For me on mac: cmake . -D USE_BACKEND=Eigen -D CMAKE_CXX_COMPILER=g++-12 -D CMAKE_C_COMPILER=gcc-12 -DUSE_AVX2=1 -DCMAKE_CXX_FLAGS='-march=native'
2. For me on linux desktop: cmake -D USE_BACKEND=CUDA -D CUDNN_INCLUDE_DIR=$CUDNN_INCLUDE_DIR -D CUDNN_LIBRARY=$CUDNN_LIBRARY/libcudnn.so -D CMAKE_CUDA_COMPILER=/usr/local/cuda-11.7/bin/nvcc
2. For me on remote: cmake . -D USE_BACKEND=CUDA -D CMAKE_CXX_COMPILER=g++ -D CMAKE_C_COMPILER=gcc
2. Or for TensorRT on remote run: cmake . -D USE_BACKEND=TensorRT -D CMAKE_CXX_COMPILER=g++ -D CMAKE_C_COMPILER=gcc -D TENSORRT_INCLUDE_DIR=/jmain02/home/J2AD008/wga37/mmp10-wga37/anaconda3/envs/thts++/tensorrt/TensorRT-8.5.1.7/include -D TENSORRT_LIBRARY=/jmain02/home/J2AD008/wga37/mmp10-wga37/anaconda3/envs/thts++/tensorrt/TensorRT-8.5.1.7/lib/libnvinfer.so
3. in external/ run: wget https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b40c256-s11840935168-d2898845681.bin.gz
2. Run make: make
3. TODO: add specifics to thts build for go things

Note:
- Building with edited cmake file won't actually be able to link, but it will build
- Don't need katago program to be built itself so this is a problem I don't need to solve



## Note on running remotely / others running:
1. Makefile might need some minor editing (e.g. when I was developing on MacOS I needed to use g++-12 rather than g++ for the gcc compiler).
2. MakefileJade is setup as the makefile on our remote server.
3. Might need to play with some of the linker flags depending on if trying to compile KataGo to run on CPU/CuDNN/TensorRT.
