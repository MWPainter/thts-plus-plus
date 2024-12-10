CXX = g++



#####
# Defining sources 
#####

SRC_DIR = src
TEST_DIR = test
BIN_DIR = bin
PY_DIR = py

SOURCES = $(wildcard src/*.cpp)
SOURCES += $(wildcard src/algorithms/*.cpp)
SOURCES += $(wildcard src/algorithms/common/*.cpp)
SOURCES += $(wildcard src/algorithms/est/*.cpp)
SOURCES += $(wildcard src/algorithms/ments/*.cpp)
SOURCES += $(wildcard src/algorithms/ments/dents/*.cpp)
SOURCES += $(wildcard src/algorithms/ments/rents/*.cpp)
SOURCES += $(wildcard src/algorithms/ments/tents/*.cpp)
SOURCES += $(wildcard src/algorithms/uct/*.cpp)
SOURCES += $(wildcard src/distributions/*.cpp)
OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(SOURCES))

TEST_SOURCES = $(wildcard test/*.cpp)
TEST_SOURCES += $(wildcard test/algorithms/*.cpp)
TEST_SOURCES += $(wildcard test/distributions/*.cpp)
TEST_SOURCES += $(wildcard test/mo/*.cpp)
TEST_OBJECTS = $(patsubst test/%.cpp, bin/test/%.o, $(TEST_SOURCES))

MO_SOURCES = $(wildcard mo/*.cpp)
MO_OBJECTS = $(patsubst mo/%.cpp, bin/mo/%.o, $(MO_SOURCES))

PY_SOURCES = $(wildcard py/*.cpp)
PY_OBJECTS = $(patsubst py/%.cpp, bin/py/%.o, $(PY_SOURCES))

PY_ENV_SERVER_MAIN = py/env_server/main.cpp
PY_ENV_SERVER_MAIN_OBJ = bin/py/env_server/main.o

PY_MAIN = py/main/module.cpp
PY_MAIN_OBJ = bin/py/main/module.o

MAIN_SOURCES = $(wildcard main/*.cpp)
MAIN_OBJECTS = $(patsubst main/%.cpp, bin/main/%.o, $(MAIN_SOURCES))

GTEST = external/googletest/build/lib/libgtest_main.a



#####
# Defining flags and targets 
#####

# Variables that need to get updated per machine
CONDA_ENV_NAME = thts++mo
PYTHON_WITH_VER = python3.12
ANACONDA_ENVS_HOME = /home/michael/anaconda3/envs
BOOST_INCLUDE_DIR = /home/michael/cpp_include

# Includes
INCLUDES = -I. -Iinclude -Isrc -Iexternal 
INCLUDES += -Iexternal/eigen 
INCLUDES += -Iexternal/qhull/src 
# INCLUDES += -Iexternal/lemon-1.3.1/build/lemon/include # no longer using lemon
INCLUDES += -Iexternal/bayesopt/include
INCLUDES += -Iexternal/clp/dist/include
INCLUDES += -I$(BOOST_INCLUDE_DIR) 

# Includes for using pybind11
INCLUDES += -Iexternal/pybind11/include $$(python -m pybind11 --includes) -Ipy
INCLUDES += -I$(ANACONDA_ENVS_HOME)/$(CONDA_ENV_NAME)/include/$(PYTHON_WITH_VER)

# Includes for tests
TEST_INCLUDES = -Iexternal/googletest/build/include

# C++ flags
CPPFLAGS = $(INCLUDES) -Wall -std=c++20 
CPPFLAGS += -O3

# C++ flags for building pybind11 executable/library
PY_LIB_CPPFLAGS += -fPIC -fvisibility=hidden # needed to create shared library
PY_EX_CPPFLAGS += -pie -fPIE # needed to create executable

# C++ flags for tests + debugging
TEST_CPPFLAGS = 
CPPFLAGS_DEBUG = -g -ggdb3

# ld flags
LDFLAGS = -Lexternal/qhull/qhull_build/lib 
# LDFLAGS += -L/usr/lib/x86_64-linux-gnu  # Where GLPK library is
# LDFLAGS += -Lexternal/lemon-1.3.1/build/lemon/lib 
LDFLAGS += -Lexternal/bayesopt/build/lib
LDFLAGS += -Lexternal/clp/dist/lib
LDFLAGS += -lqhullcpp -lqhullstatic_r 
# LDFLAGS += -lemon # no longer using lemon
# LDFLAGS += -lglpk # no longer using glpk
LDFLAGS += -lpthread 
LDFLAGS += -lbayesopt -lnlopt
LDFLAGS += -lClp -lCoinUtils

# ld flags for building with pybind11
PY_LD_LOCS =  -L$(ANACONDA_ENVS_HOME)/$(CONDA_ENV_NAME)/lib
PY_LDFLAGS = $(PY_LD_LOCS) -l$(PYTHON_WITH_VER)

# ld flags for tests
TEST_LDFLAGS = -Lexternal/googletest/build/lib 
TEST_LDFLAGS += -lgtest -lgtest_main -lgmock

# targets
TARGET_THTS = thts
TARGET_THTS_TEST = thts-test
TARGET_THTS_TEST_DEBUG = thts-test-debug
TARGET_THTS_PY_LIB = thtspp
TARGET_THTS_PY_LIB_DEBUG = thtspp-debug
TARGET_THTS_PY_EX = pyex
TARGET_THTS_PY_EX_DEBUG = pyex-debug
TARGET_MO_EXPR = moexpr
TARGET_MO_EXPR_DEBUG = moexpr-debug
TARGET_PY_ENV_SERVER = py_env_server
TARGET_PY_ENV_SERVER_DEBUG = py_env_server-debug

# python lib target
THTS_PY_LIB_FULL_NAME = thts$$(python3.12-config --extension-suffix)



#####
# Default target
#####

# Default, build everything
all: $(TARGET_THTS_PY_EX) $(TARGET_MO_EXPR) $(TARGET_PY_ENV_SERVER) $(TARGET_THTS_TEST) 



#####
# (Custom) Rules to build object files
# Adapted from: https://stackoverflow.com/questions/41568508/makefile-compile-multiple-c-file-at-once/41924169#41924169
# N.B. A rule of the form "some_dir/%.o: bin/some_dir/%.cpp" for some reason makes make recompile all files everytime
#####

# Required to enable use of $$
.SECONDEXPANSION:

## General rule formula for building object files 
## N.B. $(@D) gets the directory of the file
#$(OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
#	@mkdir -p $(@D)
#	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build object files rule
$(OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(PY_EX_CPPFLAGS) $(PY_LIB_CPPFLAGS) -c -o $@ $<

# Build python object files rule
$(PY_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(PY_EX_CPPFLAGS) $(PY_LIB_CPPFLAGS) -c -o $@ $<

# Build multi-objective object files rule
$(MO_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(PY_EX_CPPFLAGS) $(PY_LIB_CPPFLAGS) -c -o $@ $<

# Build test object files rule
$(TEST_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(PY_EX_CPPFLAGS) $(PY_LIB_CPPFLAGS) -c -o $@ $<

$(PY_MAIN_OBJ) : $(PY_MAIN)
	@mkdir -p bin/py/main
	$(CXX) $(CPPFLAGS) $(PY_EX_CPPFLAGS) $(PY_LIB_CPPFLAGS) -c -o $@ $<

$(PY_ENV_SERVER_MAIN_OBJ) : $(PY_ENV_SERVER_MAIN)
	@mkdir -p bin/py/env_server
	$(CXX) $(CPPFLAGS) $(PY_EX_CPPFLAGS) $(PY_LIB_CPPFLAGS) -c -o $@ $<

# Build main object files rule
$(MAIN_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(PY_EX_CPPFLAGS) $(PY_LIB_CPPFLAGS) -c -o $@ $<


#####
# Targets
#####

# Compiling 'thts' just builds objects for now (to be used as prereq basically)
$(TARGET_THTS): $(OBJECTS)

# Build test program
$(TARGET_THTS_TEST): INCLUDES += $(TEST_INCLUDES)
$(TARGET_THTS_TEST): CPPFLAGS += $(TEST_CPPFLAGS)
$(TARGET_THTS_TEST): LDFLAGS += $(PY_LDFLAGS)
$(TARGET_THTS_TEST): LDFLAGS += $(TEST_LDFLAGS)
$(TARGET_THTS_TEST): $(OBJECTS) $(PY_OBJECTS) $(MO_OBJECTS) $(TEST_OBJECTS)
	$(CXX) $(CPPFLAGS) $(PY_EX_CPPFLAGS) -o $@ $^ $(GTEST) $(LDFLAGS)

# Add a debug tests target. Adds -g to flags for debug info, and then just runs tests target
$(TARGET_THTS_TEST_DEBUG): CPPFLAGS += $(CPPFLAGS_DEBUG)
$(TARGET_THTS_TEST_DEBUG): $(TARGET_THTS_TEST)

# Building the python library
$(TARGET_THTS_PY_LIB): LDFLAGS += $(PY_LDFLAGS)
$(TARGET_THTS_PY_LIB): $(OBJECTS) $(PY_OBJECTS) $(MO_OBJECTS) $(PY_MAIN_OBJ)
	$(CXX) -shared $(PY_LIB_CPPFLAGS) $(CPPFLAGS) $^ -o $(THTS_PY_LIB_FULL_NAME) $(LDFLAGS)

# C++ entry point for debugging python C++ entry point
$(TARGET_THTS_PY_LIB_DEBUG): CPPFLAGS += $(CPPFLAGS_DEBUG)
$(TARGET_THTS_PY_LIB_DEBUG): $(TARGET_THTS_PY_EX)

# C++ entry point
$(TARGET_THTS_PY_EX): LDFLAGS += $(PY_LDFLAGS)
$(TARGET_THTS_PY_EX): $(OBJECTS) $(PY_OBJECTS) $(MO_OBJECTS) $(PY_MAIN_OBJ)
	$(CXX) -shared $(PY_EX_CPPFLAGS) $(CPPFLAGS) $^ -o $(TARGET_THTS_PY_EX) $(LDFLAGS)

# C++ entry point for debugging python C++ entry point
$(TARGET_THTS_PY_EX_DEBUG): CPPFLAGS += $(CPPFLAGS_DEBUG)
$(TARGET_THTS_PY_EX_DEBUG): $(TARGET_THTS_PY_EX)

# Py Env Server entry point
$(TARGET_PY_ENV_SERVER): LDFLAGS += $(PY_LDFLAGS)
$(TARGET_PY_ENV_SERVER): $(OBJECTS) $(PY_OBJECTS) $(MO_OBJECTS) $(PY_ENV_SERVER_MAIN_OBJ)
	$(CXX) -shared $(PY_EX_CPPFLAGS) $(CPPFLAGS) $^ -o $(TARGET_PY_ENV_SERVER) $(LDFLAGS)

# Debug Py Env Server entry point
$(TARGET_PY_ENV_SERVER_DEBUG): CPPFLAGS += $(CPPFLAGS_DEBUG)
$(TARGET_PY_ENV_SERVER_DEBUG): $(TARGET_PY_ENV_SERVER)

# Expr entry point
$(TARGET_MO_EXPR): LDFLAGS += $(PY_LDFLAGS)
$(TARGET_MO_EXPR): $(OBJECTS) $(PY_OBJECTS) $(MO_OBJECTS) $(MAIN_OBJECTS)
	$(CXX) -shared $(PY_EX_CPPFLAGS) $(CPPFLAGS) $^ -o $(TARGET_MO_EXPR) $(LDFLAGS)

# Debug expr entry
$(TARGET_MO_EXPR_DEBUG): CPPFLAGS += $(CPPFLAGS_DEBUG)
$(TARGET_MO_EXPR_DEBUG): $(TARGET_MO_EXPR)


#####
# Clean
#####

# Clean up compiled files
# "[X] && Y || Z" is bash for if X then Y else Z 
# [ -e <filename> ] checks if filename exists
# : is a no-op 
clean:
	@rm -rf $(BIN_DIR) > /dev/null 2> /dev/null
	@[ -f $(TARGET_THTS_TEST) ] && @rm $(TARGET_THTS_TEST) > /dev/null 2> /dev/null || :


#####
# Phony targets, so make knows when a target isn't producing a corresponding output file of same name
#####
.PHONY: clean $(TARGET_THTS) $(TARGET_THTS_TEST_DEBUG) $(TARGET_THTS_PY_LIB) $(TARGET_THTS_PY_EX_DEBUG) $(TARGET_MO_EXPR_DEBUG)
