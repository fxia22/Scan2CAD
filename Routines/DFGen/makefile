all : dfgen sdfgen

dfgen : dfgen.cpp makelevelset3.cpp
	c++ -I../Base -std=c++17 -O3 -Wall $^ -o $@

sdfgen : sdfgen.cpp makelevelset3.cpp
	c++ -I../Base -std=c++17 -O3 -Wall $^ -o $@

clean:
	rm main
