CXX = g++
CXXFLAGS = -std=gnu++17 -O3 -march=native -pthread -Wall
TARGET = TestSuite
OBJECTS = TestSuite.o ScalarParser.o ParallelParser.o
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
clean:
	rm -f $(OBJECTS) $(TARGET)