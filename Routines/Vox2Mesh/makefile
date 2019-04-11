all : vox2mesh show_vox

vox2mesh : main.cpp
	c++  -std=c++11 -I../Base -Wall -O3 -o $@ $<

show_vox : show_vox.cpp
	c++  -std=c++11 -I../Base -Wall -O3 -o $@ $<

clean:
	rm vox2mesh && rm show_vox
