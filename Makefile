CXX = g++



#####
# Defining targets, sources and flags
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
TEST_OBJECTS = $(patsubst test/%.cpp, bin/test/%.o, $(TEST_SOURCES))

# PY_MODULE_DEF = py/module/module.cpp
PY_SOURCES = $(wildcard py/*.cpp)
PY_OBJECTS = $(patsubst py/%.cpp, bin/py/%.o, $(PY_SOURCES))

GTEST = external/googletest/build/lib/libgtest_main.a

INCLUDES = -Iinclude/ -Isrc/ -Iexternal/ -I. 
TEST_INCLUDES = -Iexternal/googletest/build/include
PY_INCLUDES = -Iexternal/pybind11/include $$(python3.9 -m pybind11 --includes) -Ipy/
PY_INCLUDES += -I/home/michael/anaconda3/envs/thts++/include/python3.9
# PY_INCLUDES += -I/jmain02/home/J2AD008/wga37/mmp10-wga37/anaconda3/envs/thts++/x86_64-conda-linux-gnu/include/c++/11.2.0
# PY_INCLUDES += -I/jmain02/home/J2AD008/wga37/mmp10-wga37/anaconda3/envs/thts++/x86_64-conda-linux-gnu/sysroot/usr/include/

CPPFLAGS = $(INCLUDES) -Wall -std=c++20
PY_CPPFLAGS = -fPIC -fvisibility=hidden # needed to create shared library
PY_EX_CPPFLAGS = -pie -fPIE # needed to create executable
TEST_CPPFLAGS = 
CPPFLAGS_DEBUG = -g -ggdb

LDFLAGS = -lpthread
# PY_LD_LOCS =  -L/jmain02/home/J2AD008/wga37/mmp10-wga37/anaconda3/envs/thts++/lib 
# PY_LD_LOCS += -L/jmain02/home/J2AD008/wga37/mmp10-wga37/anaconda3/envs/thts++/x86_64-conda-linux-gnu/lib64
# PY_LDFLAGS = $(PY_LD_LOCS) -lpython3.10
PY_LD_LOCS =  -L/home/michael/anaconda3/envs/thts++/lib
PY_LDFLAGS = $(PY_LD_LOCS) -lpython3.9
TEST_LDFLAGS = -Lexternal/googletest/build/lib -lgtest -lgtest_main -lgmock

TARGET_THTS = thts
TARGET_THTS_TEST = thts-test
TARGET_THTS_TEST_DEBUG = thts-test-debug
TARGET_THTS_PY_LIB = thtspp
TARGET_THTS_PY_EX = pyex
TARGET_THTS_PY_EX_DEBUG = pyex-debug

THTS_PY_LIB_FULL_NAME = thts$$(python3.9-config --extension-suffix)



#####
# Default target
#####

# Default, build everything
all: $(TARGET_THTS_TEST)



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
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build python object files rule
$(PY_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build test object files rule
$(TEST_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<



#####
# Targets
#####

# Compiling 'thts' just builds objects for now (to be used as prereq basically)
$(TARGET_THTS): $(OBJECTS)

# Build test program
$(TARGET_THTS_TEST): INCLUDES += $(TEST_INCLUDES)
$(TARGET_THTS_TEST): CPPFLAGS += $(TEST_CPPFLAGS)
$(TARGET_THTS_TEST): LDFLAGS += $(TEST_LDFLAGS)
$(TARGET_THTS_TEST): $(OBJECTS) $(TEST_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(GTEST) $(LDFLAGS)

# Add a debug tests target. Adds -g to flags for debug info, and then just runs tests target
$(TARGET_THTS_TEST_DEBUG): CPPFLAGS += $(CPPFLAGS_DEBUG)
$(TARGET_THTS_TEST_DEBUG): $(TARGET_THTS_TEST)

# Building the python library
$(TARGET_THTS_PY_LIB): INCLUDES += $(PY_INCLUDES)
$(TARGET_THTS_PY_LIB): CPPFLAGS += $(PY_CPPFLAGS)
$(TARGET_THTS_PY_LIB): LDFLAGS += $(PY_LDFLAGS)
$(TARGET_THTS_PY_LIB): $(OBJECTS) $(PY_OBJECTS)
	$(CXX) -shared $(CPPFLAGS) $^ -o $(THTS_PY_LIB_FULL_NAME) $(LDFLAGS)

# C++ entry point
$(TARGET_THTS_PY_EX): INCLUDES += $(PY_INCLUDES)
$(TARGET_THTS_PY_EX): CPPFLAGS += $(PY_CPPFLAGS)
$(TARGET_THTS_PY_EX): CPPFLAGS += $(PY_EX_CPPFLAGS)
$(TARGET_THTS_PY_EX): LDFLAGS += $(PY_LDFLAGS)
$(TARGET_THTS_PY_EX): $(OBJECTS) $(PY_OBJECTS)
	$(CXX) -shared $(CPPFLAGS) $^ -o $(TARGET_THTS_PY_EX) $(LDFLAGS)

# C++ entry point for debugging python library
$(TARGET_THTS_PY_EX_DEBUG): CPPFLAGS += $(CPPFLAGS_DEBUG)
$(TARGET_THTS_PY_EX_DEBUG): $(TARGET_THTS_PY_EX)



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
.PHONY: clean $(TARGET_THTS) $(TARGET_THTS_TEST_DEBUG) $(TARGET_THTS_PY_LIB) $(TARGET_THTS_PY_EX_DEBUG)
