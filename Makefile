TARGET_EXEC ?= mcwf.out
TARGET_LIB ?= mcwf.so

BUILD_DIR ?= ./build
SRC_DIRS ?= ./src

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d) ./include/eigen /usr/lib/python3.8/site-packages/pybind11/include /usr/include/python3.8
INC_FLAGS := $(addprefix -isystem ,$(INC_DIRS))

CXX = g++  -fPIC
CPPFLAGS ?= $(INC_FLAGS) -MMD -MP -O3 -march=native -g3 -fno-omit-frame-pointer -fopenmp
CPPFLAGS += -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-promo -Wstrict-null-sentinel -Wswitch-default -Wundef  -Wno-unused #-Wsign-conversion -Werror -Wstrict-overflow=5
LDFLAGS += -lyaml-cpp -fopenmp

NOLINKOBJ = $(filter-out ./build/./src/PythonBindings.cpp.o, $(OBJS))

all: $(TARGET_LIB) $(TARGET_EXEC)

$(TARGET_LIB): $(OBJS)
	$(CXX) -shared -Wl,-soname,$@ -o $@ $(OBJS) $(LDFLAGS)

$(TARGET_EXEC): $(NOLINKOBJ)
	$(CXX) $(NOLINKOBJ) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


.PHONY: clean all

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p
