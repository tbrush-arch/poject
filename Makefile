CXX = g++
# 1. Define the path to the Highway library root
HWY_DIR = google/highway/highway-224b014b1e6ebd1b9c1e134ebb5fbce899844c79/

# 2. Add -I$(HWY_DIR) to CXXFLAGS so the compiler finds "hwy/..."
CXXFLAGS = -std=gnu++17 -O3 -march=native -pthread -Wall -I$(HWY_DIR)
TARGET = TestSuite

# 3. Include required Highway source objects for linking
HWY_OBJS = $(HWY_DIR)hwy/targets.o $(HWY_DIR)hwy/abort.o $(HWY_DIR)hwy/per_target.o
OBJECTS = TestSuite.o ScalarParser.o ParallelParser.o $(HWY_OBJS)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Standard rule for your project files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 4. Add a rule to compile the Highway .cc files
$(HWY_DIR)hwy/%.o: $(HWY_DIR)hwy/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
