CXX = g++



#####
# Defining targets, sources and flags
#####

SRC_DIR = src
TEST_DIR = test
BIN_DIR = bin

SOURCES = $(wildcard src/*.cpp)
SOURCES += $(wildcard src/algorithms/*.cpp)
SOURCES += $(wildcard src/algorithms/common/*.cpp)
SOURCES += $(wildcard src/algorithms/est/*.cpp)
SOURCES += $(wildcard src/algorithms/ments/*.cpp)
SOURCES += $(wildcard src/algorithms/ments/dents/*.cpp)
SOURCES += $(wildcard src/algorithms/ments/rents/*.cpp)
SOURCES += $(wildcard src/algorithms/ments/tents/*.cpp)
SOURCES += $(wildcard src/algorithms/uct/*.cpp)
OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(SOURCES))
TEST_SOURCES = $(wildcard test/*.cpp)
TEST_SOURCES += $(wildcard test/algorithms/*.cpp)
TEST_OBJECTS = $(patsubst test/%.cpp, bin/test/%.o, $(TEST_SOURCES))

GTEST = external/googletest/build/lib/libgtest_main.a

INCLUDES = -Iinclude/ -Isrc/ -Iexternal/ -I.
TEST_INCLUDES = -Iexternal/googletest/build/include

CPPFLAGS = $(INCLUDES) -Wall -std=c++17
TEST_CPPFLAGS = 
CPPFLAGS_DEBUG = -g

LDFLAGS = -lpthread
TEST_LDFLAGS = -Lexternal/googletest/build/lib -lgtest -lgtest_main -lgmock

TARGET_THTS = thts
TARGET_THTS_TEST = thts-test
TARGET_THTS_TEST_DEBUG = thts-test-debug



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
	echo $(TEST_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(GTEST) $(LDFLAGS)

# Add a debug tests target. Adds -g to flags for debug info, and then just runs tests target
$(TARGET_THTS_TEST_DEBUG): CPPFLAGS += $(CPPFLAGS_DEBUG)
$(TARGET_THTS_TEST_DEBUG): $(TARGET_THTS_TEST)



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
.PHONY: clean $(TARGET_THTS) $(TARGET_THTS_TEST_DEBUG)
