# Simplex Maps For Multi-Objective Monte Carlo Tree Search (aka the `xpr_mo` branch of THTS++)

Branch forks from `xprmntl_mo` which implements the Multi-Objective algorithms, this branch adds and contains experiments for the Multi-Objective algorithms

Additionally, this will contain replicated experiments from [Convex Hull Monte Carlo Tree Search](https://arxiv.org/abs/2003.04445) for my thesis plots

Main readme is below


# thts-plus-plus
THTS Implementation in C++, with Python bindings (eventually). By default, running `make` will just compile unit tests to an executable called `thts-test`, if you run this some tests will fail but they should all contain `todo` in their names, as they are mostly placeholder tests.



## Code Overview

All subdirectories in the `include/` folder contain `README.md` files, which give an overview of the code from that 
directory. Currently a work in progress, so lots of barebone notes rather than documentation (tends to get more bare 
the deeper into directories you get). 

Also code includes pretty heavy commenting, so hopefully most functions have descriptive enough headers to know how 
to use and interact with the classes. Additionally, plenty of functions have comments in the implementation files 



## Compiling and Running 

To compile code, will need to update some variables in the makefile for each machine to point at the right things.

To run the code, also need to update variables in `export_local_paths.sh` appropriately and 
run `source export_local_paths.sh` to setup the `PYTHONPATH` and `LD_LIBRARY_PATH` environment variables appropriately. 

TODO: would like to try and automate this in the future if possible?



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



## Installing pybind11
This step should have been done already for gtest, so can be skipped (cloneing submodules):
```
git submodule init
git submodule update
```

Should be able to copy development setup with
```
conda create -n thts++mo python=3.12
pip install -r requirements.txt
```

#### Previous notes on installing pybind were:

Then we're using Anaconda3 to create a python virtual environment. Assuming we already have Anaconda3 installed, 
make a new virtual env called `thts++` with:
```
conda create -n thts++ python=3.9
```

And install pybind11 with:
```
conda install -c conda-forge pybind11
conda install pytest
```

Then check that the C++ library is working as intended by building and running tests (pybind11 is a header only 
library, so doesn't need to be compiled to be used):
```
cd external/pybind11
mkdir build
cd build
cmake ..
make check -j 4
```



## Installing eigen

Header only library, nothing to be done




## Installing qhull

run `make` in `external/qhull`



## Installing CLP

Was using lemon with GLPK (below) but some of the linear programs were being returned infeasible when feasible, and 
couldn't replicate the issue to properly debug (returned feasible when trying to replicate). So going to move to CLP 
from Coin-OR, as it seems better/more recently maintained.

I followed the `coinbrew` installation from https://github.com/coin-or/Clp, starting from this directory:
```
cd external
mkdir clp
cd clp

wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
chmod u+x coinbrew
./coinbrew fetch Clp@master
./coinbrew build Clp
``` 



## ~~Installing lemon~~

~~For the LP solver, we use lemon https://lemon.cs.elte.hu/pub/tutorial/a00020.html, because it provides a really 
clean symbolic interface to use. Really its a wrapper around other packages such as GLPK, Clp, Cbc, ILOG CPLEX 
and SoPlex https://lemon.cs.elte.hu/trac/lemon/wiki/InstallLinux. So, it gives a clean and easy interface, to other 
packages that have already done a lot of hard coding work.~~

~~Originally tried google's OR tools, but the code base was huge, more fiddly to work with and get running. Google's 
OR tools provides a really nice python interface, which is similar to Lemon's, but the C++ interface is a lot uglier, 
hence why moving to lemon after some more research.~~

~~First make sure that `glpk` or your linear program solver of choice (which is compatible with lemon) is installed. To 
install `glpk` I had to run:~~
```
sudo apt-get install glpk-doc glpk-utils libglpk-dev libglpk40
```

~~To install lemon (which cmake will find glpk or the other linear solver installed), change to the lemon directory, 
make a build folder and run the cmake and make commands:~~
```
cd external/lemon-1.3.1
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=lemon ..
make
make install
```




## Installing BayesOpt

Follow BayesOpt install guide (mostly, see below), dont need the python or octave stuff, as doing C++: 
https://rmcantin.github.io/bayesopt/html/install.html

Turns out apt-get'ing boost libraries downloads v1.74, which is a version where trying to create a vector will cause 
compiler errors... Tried doing ppa stuff to get an updated version but it just broke my apt-get. Resorted to 
downloading lastest tar.gz from here: https://www.boost.org/users/history/version_1_86_0.html, then unpacking and 
copying the files inside the "boost" folder to "/usr/include/boost" where it would be installed anyway. Going to 
have to remember that if ever want to remove/update it, that need to delete this folder manually. And also it doesn't 
need compiling because boost is mostly header only (appart from a few parts which I hope we're not using or ever going 
to use).

I also like having the build directory and not installing globally, so did:
```
cd external/bayesopt
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=.
make install
```

I also added the line `set(CMAKE_CXX_STANDARD 20) # thts++ change: updated standard to c++20` to the bayesopt `cmakelists.txt`

Wasnt so simple getting it to work on jade, needed the entire library, not just the include (header) files, and had to install fresh rather than copying header files. Not sure why.




## Changelog

This is a space to say what's included and updated in any versioning happening. I'll create the v0.1 branch when the 
repository is in a state where I feel comfortable handing it over to other people. v1.0 are things that I would like to 
have in some sort of 'full release', but who know's if we'll ever get to that point :)

The plan for the branch structure is as follows:
- `xpr_*` denotes a branch that was created to run experiments for a project
    - these are unsupported and left as is
    - I wont be deleting these branches
- `xprmntl_*` denotes a branch containing code that is in development
    - the aim is to eventually make a pull request into main once development and testing is completed
    - these will be deleted after successfully merging them into main
- `vX.Y` denotes a release branch, that should hopefully be somewhat stable, for version X.Y
    - the aim is that any bugfixes pushed to `main` will also be pushed to `vX.Y`, while it is the most up to date version
    - when a branch containing a newer version is created, we wont wont update this branch anymore, as we will be updating that newer release version instead (at least while I'm a one man team working on this)
- `vX.Y_unsupported_*` denotes an unsupported release branch
    - unsupported release branches contain work that I didn't want to throw away, but there is no immediatelly forseeable use case for them. I want to keep them around in case I am wrong, and there is a case that I did not forsee where the code is actually useful
- `main` denotes the bleeding edge branch
    - this is still intended to be stable, as any unstable work should be in a `xprmntl_*` branch
    - however if we want to release features A, B, C together, we might push A into main when it is finished, before B and C are completed for example


### v0.1_unsupported_convex_hull

The algorithms using convex hull logic appear to be superceeded by the contextual zooming and algorithms using simplex maps, and the logic for convex hulls and pareto fronts provide a lot of code bloat. Hence this work will be put into an unsupported branch at v0.1 release.

### to add to v0.1 changelog 

- things added for python (python interface, and running python stuff from C++)
- Multi-Objective algorithms
    - Add simplex maps + contextual zooming as things that people might actually want to use
    - Make an v0.1_unsupported_rents_and_tents branch too, like v0.1_unsupported_convex_hull

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
something that could be improved, or even just better explained/documented please add your own issues, I am very keen for this library to be easy to use for other people.
