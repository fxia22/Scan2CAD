all : sdf2vox

sdf2vox : main.cpp
	c++ -I../Base -std=c++11 -Wall -O3 -o $@ $<

clean:
	rm sdf2vox
