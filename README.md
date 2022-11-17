# thts-plus-plus
THTS Implementation in C++ with Python bindings

## Using gtest
1. clone submodule
2. cd external/googletest
3. mkdir build
4. cd build
5. cmake ../ -DCMAKE_INSTALL_PREFIX=. -DCMAKE_C_COMPILER=<your_c_compiler> -DCMAKE_CXX_COMPILER=<your_cxx_compiler>
6. cd ..
7. make
8. make install

## TODO
1. Setup testing suite
2. Port code and test as go
3. Tutorial (markdown, w/ diagrams)
4. Python bindings, and being able to extend/use in python 
5. Jupyter notebook tutorial
6. Continuous integration tools
