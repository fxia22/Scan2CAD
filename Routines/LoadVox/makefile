all : LoadVox.so

LoadVox.so : LoadVox.cpp
	c++ -I../Base -O3 -Wall -shared -std=c++11 -fPIC `python3.6 -m pybind11 --includes` $^ -o $@

clean:
	rm LoadVox.so
