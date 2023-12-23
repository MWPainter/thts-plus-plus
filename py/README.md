# Code Overview - py/

This folder contains all of the logic to interface between C++ and python. That is:
1. exposing THTS++ library functions to python
2. allowing python environments to be used in THTS++


## module/module.cpp

The file defining the THTS++ python module

## py_thts_types.h

Defines wrappers around py::object (pybind11's python object type) to use them as thts::State, thts::Action and 
thts::Observation types.

## py_thts_env_template.py

A base class to use to define a thts environment in python

## test_env.h / test_env.py

Defines a very simple gridworld env to test with (in c++/python respectively)