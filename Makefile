CXX = g++

SOURCES = $(wildcard src/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
TESTS = $(wildcard test/*.cpp)
TESTS_OBJECTS = $(TESTS:.cpp=.o)

INCLUDES= -Iinclude/ -Isrc/

CPPFLAGS = $(INCLUDES) -Wall -std=c++17 
CPPFLAGS_DEBUG = -g
LDFLAGS = 

TARGET_THTS = thts
TARGET_THTS_TESTS = thts-tests
TARGET_THTS_TESTS_DEBUG = thts-tests-debug

all: $(TARGET_THTS_TESTS)

$(TARGET_THTS): $(OBJECTS)

$(TARGET_THTS_TESTS): $(OBJECTS) $(TESTS_OBJECTS)
	@mkdir -p build/test	
	$(CXX) $(CPPFLAGS) $(LDFLAGS) -o $@ $^

# Add a debug tests target. Adds -g to flags for debug info, and then just runs tests target
$(TARGET_THTS_TESTS_DEBUG): CPPFLAGS += $(CPPFLAGS_DEBUG)
$(TARGET_THTS_TESTS_DEBUG): $(TARGET_THTS_TESTS)

clean: 
	@rm -rf build
	@rm $(OBJECTS) $(TESTS_OBJECTS) > /dev/null
	@rm $(TARGET_THTS_TESTS) > /dev/null

.PHONY: clean $(TARGET_THTS) $(TARGET_THTS_TESTS)
