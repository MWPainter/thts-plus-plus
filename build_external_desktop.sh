dirname=$(readlink -f -- $(dirname "${BASH_SOURCE[0]}"))
export Boost_INCLUDE_DIR=/home/michael/cpp_include

cd $dirname/external/googletest
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=. -DCMAKE_CXX_FLAGS='-fPIE'
make
make install

cd $dirname/external/qhull
mkdir qhull_build
cd qhull_build
cmake .. -DCMAKE_INSTALL_PREFIX=. -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_FLAGS='-O2'
make install

cd $dirname/external
mkdir clp
cd clp
wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
chmod u+x coinbrew
./coinbrew fetch Clp@master
./coinbrew build Clp
./coinbrew build Clp ADD_CXXFLAGS="-O3 -march=native -DNDEBUG" ADD_CFLAGS="-O3 -march=native -DNDEBUG"

cd $dirname/external/bayesopt
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=. -DBoost_INCLUDE_DIR=$Boost_INCLUDE_DIR -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_FLAGS='-O3'
make install