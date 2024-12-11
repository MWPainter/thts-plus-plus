# Monte Carlo Tree Search with Boltmann Exploration

This branch (`xpr_go`) branches off the main branch at some point (before a major refactor of THTS++ `main` branch). It contains the experiments for the paper [Monte Carlo Tree Search with Boltmann Exploration](https://arxiv.org/abs/2404.07732), which was accepted at NeurIPS2023.

Below is the original readme for this branch during development.


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




# Go Stuff

To build, do the following steps:
1. in external/ directory run the following: wget https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b40c256-s11840935168-d2898845681.bin.gz
2. Run cmake in external/KataGo/cpp: cmake . -D USE_BACKEND=<your_backend_eg_CUDA> -D CMAKE_CXX_COMPILER=<your_cxx_compiler> -D CMAKE_C_COMPILER=<your_c_compiler>
    1. optional on cpu: -DUSE_AVX2=1 -DCMAKE_CXX_FLAGS='-march=native'
    2. For me on mac: cmake . -D USE_BACKEND=Eigen -D CMAKE_CXX_COMPILER=g++-12 -D CMAKE_C_COMPILER=gcc-12 -DUSE_AVX2=1 -DCMAKE_CXX_FLAGS='-march=native'
    3. For me on linux desktop: cmake -D USE_BACKEND=CUDA -D CUDNN_INCLUDE_DIR=$CUDNN_INCLUDE_DIR -D CUDNN_LIBRARY=$CUDNN_LIBRARY/libcudnn.so -D CMAKE_CUDA_COMPILER=/usr/local/cuda-11.7/bin/nvcc
    4. For me on remote: cmake . -D USE_BACKEND=CUDA -D CMAKE_CXX_COMPILER=g++ -D CMAKE_C_COMPILER=gcc
    5. Or for TensorRT on remote run: cmake . -D USE_BACKEND=TensorRT -D CMAKE_CXX_COMPILER=g++ -D CMAKE_C_COMPILER=gcc -D TENSORRT_INCLUDE_DIR=/jmain02/home/J2AD008/wga37/mmp10-wga37/anaconda3/envs/thts++/tensorrt/TensorRT-8.5.1.7/include -D TENSORRT_LIBRARY=/jmain02/home/J2AD008/wga37/mmp10-wga37/anaconda3/envs/thts++/tensorrt/TensorRT-8.5.1.7/lib/libnvinfer.so
3. Run make: make
4. Now should be able to build make targets THTS++


Note:
- Building with edited cmake file won't actually be able to link, but it will build
- Don't need katago program to be built itself so this is a problem I don't need to solve



## Note on running remotely / others running:
1. Makefile might need some minor editing (e.g. when I was developing on MacOS I needed to use g++-12 rather than g++ for the gcc compiler).
2. MakefileJade is setup as the makefile on our remote server.
3. Might need to play with some of the linker flags depending on if trying to compile KataGo to run on CPU/CuDNN/TensorRT.







# Supplementary Material Submission Section

Here's the readme for supplemental material submission. The following changes to the repo are made:
1. All other readme's deleted, including this section in this readme and everything above it
2. All git files removes
3. Makefile and MakefileJade deleted, and MakefileSupp renamed to Makefile
4. Removed any local files (the katago network, the results and plot folders)





# Instructions for Compiling (Supplementary Material)

To build KataGo:
1. in `external/` directory run the following: `wget https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b40c256-s11840935168-d2898845681.bin.gz`
2. Run `cmake` in `external/KataGo/cpp`, consider the following options
    1. Cuda backend: `cmake . -D USE_BACKEND=CUDA`
    2. Faster option on cpu:  `cmake . -D USE_BACKEND=Eigen -DUSE_AVX2=1 -DCMAKE_CXX_FLAGS='-march=native`
    3. Standard cpu option: `cmake . -D USE_BACKEND=Eigen`
3. Run make: `make`

To build programs:
1. `make go` will build the `go` program, that runs go games
2. `make go-cuda` will build the `go` program using the cuda backend, you should use this if you built KataGo with the cuda backend
3. `make thts-run-toy-env` will build the `thts-run-toy-env` program, that runs the MCTS algorithms on gridworld environments

Running:
1. If you just want to run a game of go:
    1. `./go 000_debug` (this will run a game between PUCT as black, and BST as white)
2. If you want to run games of go between two specific algorithms:
    1. `./go 101_round_robin_9x9 <alg_for_black> <alg_for_white>` where options for algorithms are:
        1. `kata` (PUCT)
        2. `uct`
        3. `puct`
        4. `ments`
        5. `dents`
        6. `rents`
        7. `tents`
        8. `est` (an old name for BST) 
        9. `dents`
3. If you want to run MCTS algorithms on the gridworld environments:
    1. `./thts-run-toy-env <exprid>`
        1. `<exprid>` is one of the following:
            1. `051_fl12_hps` for the Frozen Lake hyper-parameter search
            2. `052_fl12_test` for the Frozen Lake test environment
            3. `091_s6_hps` for the Sailing problem hyper-parameter search
            3. `092_s6_hps` for the Sailing problem test environment

Looking at results:
1. Results for go games run can be found in the `results/go/<exprid>` folder where exprid is either 000_debug, or 101_round_robin_9x9
    1. `match_x.csv` files contains a log for game number x
    2. `results.csv` files contain the cumulative result of the match, over all games (sorry for confusing naming of files!)
2. `python produce_go_graphics.py 9 1 results/go/<exprid>`
    1. Will produce images for the games run, where `<exprid>` is either `000_debug`, or `101_round_robin_9x9` depending on what you ran the `./go` program with
    2. Note that this requires the PyCairo package
    3. Images can be found in the same folder as the `match_x.csv` files
3. `python plot.py <exprid>`
    1. Will produce plots for gridworld experiments (will make a plots folder)
    2. Requires matplotlib and seaborn
    3. `<exprid>` should match the `<exprid>` you ran `./thts-run-toy-env` with
