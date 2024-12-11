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
SOURCES += $(wildcard src/toy_envs/*.cpp)
OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(SOURCES))
TEST_SOURCES = $(wildcard test/*.cpp)
TEST_SOURCES += $(wildcard test/algorithms/*.cpp)
TEST_SOURCES += $(wildcard test/toy_envs/*.cpp)
TEST_OBJECTS = $(patsubst test/%.cpp, bin/test/%.o, $(TEST_SOURCES))

RUN_TOY_ENVS_MAIN_SRC = main/main_toy.cpp
RUN_TOY_ENVS_MAIN_OBJ = $(BIN_DIR)/main/main_toy.o

GTEST = external/googletest/build/lib/libgtest_main.a

INCLUDES = -Iinclude/ -Isrc/ -Iexternal/ -I.
TEST_INCLUDES = -Iexternal/googletest/build/include

CPPFLAGS = $(INCLUDES) -Wall -std=c++17 -O2
TEST_CPPFLAGS = 
CPPFLAGS_DEBUG = -g

LDFLAGS = -L/jmain02/home/J2AD008/wga37/mmp10-wga37/anaconda3/envs/thts++/lib -lpthread #-ltcmalloc 
TEST_LDFLAGS = -Lexternal/googletest/build/lib -lgtest -lgtest_main -lgmock

TARGET_THTS = thts
TARGET_THTS_TEST = thts-test
TARGET_THTS_TEST_DEBUG = thts-test-debug

TARGET_RUN_TOY_ENVS = thts-run-toy-env
TARGET_RUN_TOY_ENVS_DEBUG = thts-run-toy-env-debug



#####
# Go target names
#####

# Recursive wildcard from: https://stackoverflow.com/questions/2483182/recursive-wildcards-in-gnu-make/18258352#18258352
rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

# Get all katago object files using rwildcard (and remove the main, so don't find two main functions when compiling)
# But need to replace main with fake main, because main.h is imported in other files..... urgh
KATAGO_FAKE_MAIN_SOURCE = external/KataGo/cpp/main_fake.cpp
KATAGO_FAKE_MAIN_OBJECT = external/KataGo/cpp/main_fake.o
KATAGO_OBJECTS_INCL_MAIN = $(call rwildcard,external/KataGo/cpp/CMakeFiles/katago.dir,*.o)
KATAGO_OBJECTS = $(patsubst %main.cpp.o, , $(KATAGO_OBJECTS_INCL_MAIN))

GO_SOURCES = $(wildcard src/go/*.cpp)
GO_OBJECTS = $(patsubst src/go/%.cpp, $(BIN_DIR)/src/go/%.o, $(GO_SOURCES))
GO_TEST_SOURCES = $(wildcard test/go/*.cpp)
GO_TEST_OBJECTS = $(patsubst test/go/%.cpp, $(BIN_DIR)/test/go/%.o, $(GO_TEST_SOURCES))

GO_MAIN_SRC = main/main_go.cpp
GO_MAIN_OBJ = $(BIN_DIR)/main/main_go.o

GO_INCLUDES = -Iexternal/eigen/

GO_CPPFLAGS = $(GO_INCLUDES)

GO_CUDA_LDFLAGS = -lOpenCL #-lcudart -lcudnn -lcublas 
GO_TENSORRT_LDFLAGS = #-lnvinfer
GO_LDFLAGS = -L/jmain02/apps/cuda/11.2/lib64 -L/jmain02/home/J2AD008/wga37/mmp10-wga37/anaconda3/envs/thts++/tensorrt/TensorRT-8.5.1.7/lib -lzip -lz $(GO_CUDA_LDFLAGS) $(GO_TENSORRT_LDFLAGS)

TARGET_GO_TEST = go-test

TARGET_GO_RUN = goooo
TARGET_GO_DEBUG = go-debug



#####
# Default target
#####

# Default, build everything
all: $(TARGET_THTS_TEST) $(TARGET_RUN_TOY_ENVS) $(TARGET_GO_TEST) $(TARGET_GO_RUN)



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

# Build rule for main toy
$(RUN_TOY_ENVS_MAIN_OBJ): $(RUN_TOY_ENVS_MAIN_SRC)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<



#####
# Targets (non go stuff)
#####

# Compiling 'thts' just builds objects for now (to be used as prereq basically)
$(TARGET_THTS): $(OBJECTS)

# Build target to run thts on toy envs
$(TARGET_RUN_TOY_ENVS): $(OBJECTS) $(RUN_TOY_ENVS_MAIN_OBJ)
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -o $@ $^ 

# Debug toy envs
$(TARGET_RUN_TOY_ENVS_DEBUG): CPPFLAGS += $(CPPFLAGS_DEBUG)
$(TARGET_RUN_TOY_ENVS_DEBUG): $(TARGET_RUN_TOY_ENVS)

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
# Go build object files
#####

# Rule for building main_Fake
$(KATAGO_FAKE_MAIN_OBJECT) : $(KATAGO_FAKE_MAIN_SOURCE)
	$(CXX) $(CPPFLAGS) $(GO_CPPFLAGS) -c -o $@ $<

# Rule for making go main file
$(GO_MAIN_OBJ): $(GO_MAIN_SRC)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Rule for building go files
$(GO_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Rule for building go test files
$(GO_TEST_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) -c -o $@ $<



#####
# Go build targets
#####

# Build go test
$(TARGET_GO_TEST) : $(OBJECTS) $(GO_OBJECTS) $(GO_TEST_OBJECTS) $(KATAGO_FAKE_MAIN_OBJECT)
	$(CXX) $(CPPFLAGS) $(GO_CPPFLAGS) -o $@ $^ $(KATAGO_OBJECTS) $(LDFLAGS) $(GO_LDFLAGS)

# Build go expr target
$(TARGET_GO_RUN) : $(OBJECTS) $(GO_OBJECTS) $(KATAGO_FAKE_MAIN_OBJECT) $(GO_MAIN_OBJ)
	$(CXX) $(CPPFLAGS) $(GO_CPPFLAGS) -o $@ $^ $(KATAGO_OBJECTS) $(LDFLAGS) $(GO_LDFLAGS)

# Debug go target
$(TARGET_GO_DEBUG): CPPFLAGS += $(CPPFLAGS_DEBUG)
$(TARGET_GO_DEBUG): $(TARGET_GO_RUN)



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
	@[ -f $(TARGET_RUN_TOY_ENVS) ] && @rm $(TARGET_RUN_TOY_ENVS) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_RUN_TOY_ENVS) ] && @rm $(TARGET_RUN_TOY_ENVS) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_GO_TEST) ] && @rm $(TARGET_GO_TEST) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_GO_RUN) ] && @rm $(TARGET_GO_RUN) > /dev/null 2> /dev/null || :


#####
# Phony targets, so make knows when a target isn't producing a corresponding output file of same name
#####
.PHONY: clean $(TARGET_THTS) $(TARGET_THTS_TEST_DEBUG) $(TARGET_RUN_TOY_ENVS_DEBUG)














