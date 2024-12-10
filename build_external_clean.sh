dirname=$(readlink -f -- $(dirname "${BASH_SOURCE[0]}"))

cd $dirname/external/googletest
rm -rf build
cd $dirname/external/qhull
rm -rf qhull_build
cd $dirname/external
rm -rf clp
cd $dirname/external/bayesopt
rm -rf build