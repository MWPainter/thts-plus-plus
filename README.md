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

### Todos

Moved all todos to github issues, most of them are marked with v0.1 and v1.0. If you are using this repo and see 
something that could be improved, or even just better explained/documented please add your own issues (or message me, 
but I'm pretty bad at responding), I am very keen for this library to be easy to use for other people.
