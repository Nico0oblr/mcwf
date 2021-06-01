TARGET_EXEC ?= mcwf.out
TARGET_LIB ?= mcwf.so

BUILD_DIR ?= ./build
SRC_DIRS ?= ./src

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d) ./include/eigen /opt/homebrew/include
PYTHON_INC := $(shell python3 -m pybind11 --includes | sed -En "s/-I/-isystem /gp")
INC_FLAGS := $(addprefix -isystem ,$(INC_DIRS)) $(PYTHON_INC)

CXX = clang++ -arch arm64 -fPIC -Xpreprocessor -fopenmp -flto
CPPFLAGS ?= $(INC_FLAGS) -MMD -MP -O3 -g3 -fno-omit-frame-pointer -std=c++17
CPPFLAGS += $(shell python3-config --cflags | sed -En "s/-I/-isystem /gp")
CPPFLAGS += -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-promo -Wstrict-null-sentinel -Wswitch-default -Wundef  -Wno-unused
LDFLAGS +=  -L/opt/homebrew/lib -lyaml-cpp -lomp
LDFLAGS +=  $(shell python3-config --ldflags) -lpython3.9

NOLINKOBJ = $(filter-out ./build/./src/PythonBindings.cpp.o, $(OBJS))

all: $(TARGET_LIB) $(TARGET_EXEC)

$(TARGET_LIB): $(OBJS)
	$(CXX) -shared -Wl,-install_name,$@ -o $@ $(OBJS) $(LDFLAGS)

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
