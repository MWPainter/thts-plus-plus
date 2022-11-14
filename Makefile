CXX = g++-12



BINDIR = bin

SOURCES = $(wildcard src/*.cpp)
SOURCES += $(wildcard src/algorithms/*.cpp)
OBJECTS = $(patsubst src/%.cpp, $(BINDIR)/%.o, $(SOURCES))
#OBJECTS += $(patsubst src/algorithms/%.cpp, $(BINDIR)/algorithms/%.o, $(SOURCES))
TESTS = $(wildcard test/*.cpp)
TESTS += $(wildcard test/algorithms/*.cpp)
TEST_OBJECTS = $(patsubst test/%.cpp, $(BINDIR)/test/%.o, $(TESTS))
#TEST_OBJECTS += $(patsubst test/algorithms/%.cpp, $(BINDIR)/test/algorithms/%.o, $(TESTS))

INCLUDES= -Iinclude/ -Isrc/ -Iexternal/ -I.

CPPFLAGS = $(INCLUDES) -Wall -std=c++17
CPPFLAGS_DEBUG = -g

LDFLAGS =

TARGET_THTS = thts
TARGET_THTS_TEST = thts-test
TARGET_THTS_TEST_DEBUG = thts-test-debug



# Default, build everything
all: $(TARGET_THTS_TEST)

# Rule to make sure all build directories exist
bin-exists:
	@mkdir -p $(BINDIR)/test/algorithms
	@mkdir -p $(BINDIR)/algorithms

# compiling source files
$(BINDIR)/%.o : src/%.cpp bin-exists
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# compiling test source files
$(BINDIR)/test/%.o : test/%.cpp bin-exists
	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Compiling 'thts' just builds objects for now
$(TARGET_THTS): $(OBJECTS)

# Build test program
$(TARGET_THTS_TEST): $(OBJECTS) $(TEST_OBJECTS)
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -o $@ $^

# Add a debug tests target. Adds -g to flags for debug info, and then just runs tests target
$(TARGET_THTS_TEST_DEBUG): CPPFLAGS += $(CPPFLAGS_DEBUG)
$(TARGET_THTS_TEST_DEBUG): $(TARGET_THTS_TESTS)

# Clean up compiled files
clean:
	@rm -rf $(BINDIR) > /dev/null 2> /dev/null
	@rm $(TARGET_THTS_TEST) > /dev/null 2> /dev/null



.PHONY: clean $(TARGET_THTS) $(TARGET_THTS_TEST)
