CXX = g++-12



BINDIR = bin

SOURCES = $(wildcard src/*.cpp)
SOURCES += $(wildcard src/algorithms/*.cpp)
SOURCES += $(wildcard src/algorithms/common/*.cpp)
OBJECTS = $(patsubst src/%.cpp, $(BINDIR)/%.o, $(SOURCES))
TESTS = $(wildcard test/*.cpp)
TESTS += $(wildcard test/algorithms/*.cpp)
TEST_OBJECTS = $(patsubst test/%.cpp, $(BINDIR)/test/%.o, $(TESTS))

GTEST = external/googletest/build/lib/libgtest_main.a

INCLUDES = -Iinclude/ -Isrc/ -Iexternal/ -I.
TEST_INCLUDES = -Iexternal/googletest/build/include

CPPFLAGS = $(INCLUDES) -Wall -std=c++17
TEST_CPPFLAGS = 
CPPFLAGS_DEBUG = -g

LDFLAGS =
TEST_LDFLAGS = -Lexternal/googletest/build/lib -lgtest -lgtest_main -lgmock

TARGET_THTS = thts
TARGET_THTS_TEST = thts-test
TARGET_THTS_TEST_DEBUG = thts-test-debug



# Default, build everything
all: $(TARGET_THTS_TEST)

# Rule to make sure all build directories exist
bin-exists:
	@mkdir -p $(BINDIR)/test/algorithms
	@mkdir -p $(BINDIR)/algorithms/common

# compiling source files
$(BINDIR)/%.o : src/%.cpp bin-exists
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# compiling test source files
$(BINDIR)/test/%.o : test/%.cpp bin-exists
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Compiling 'thts' just builds objects for now
$(TARGET_THTS): $(OBJECTS)

# Build test program
$(TARGET_THTS_TEST): INCLUDES += $(TEST_INCLUDES)
$(TARGET_THTS_TEST): CPPFLAGS += $(TEST_CPPFLAGS)
$(TARGET_THTS_TEST): LDFLAGS += $(TEST_LDFLAGS)
$(TARGET_THTS_TEST): $(OBJECTS) $(TEST_OBJECTS)
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -o $@ $^ $(GTEST)

# Add a debug tests target. Adds -g to flags for debug info, and then just runs tests target
$(TARGET_THTS_TEST_DEBUG): CPPFLAGS += $(CPPFLAGS_DEBUG)
$(TARGET_THTS_TEST_DEBUG): $(TARGET_THTS_TEST)

# Clean up compiled files
clean:
	@rm -rf $(BINDIR) > /dev/null 2> /dev/null
	@rm $(TARGET_THTS_TEST) > /dev/null 2> /dev/null



.PHONY: clean bin-exists $(TARGET_THTS)
