#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include <unordered_map>
#include <iostream>

#include <eigen3/Eigen/Dense>
#include "Box3D.h"
#include "LoaderVOX.h"
#include "LoaderMesh.h"
#include "SE3.h"
#include "Colormap.h"
#include "args.hxx"

struct InputArgs {
	std::string in;
} inargs;



void parse_args(int argc, char** argv) {
	args::ArgumentParser parser("This is a test program.", "This goes after the options.");
	args::Group allgroup(parser, "", args::Group::Validators::All);

	args::ValueFlag<std::string> in(allgroup, "bunny.vox", "vox file", {"in"});
	

	try {
		parser.ParseCLI(argc, argv);
	} catch (args::ParseError e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		exit(1);
	} catch (args::ValidationError e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		exit(1);
	}

	inargs.in = args::get(in);

};

int main(int argc, char** argv) {
   	parse_args(argc, argv); 

	Vox vox;

	vox = load_vox(inargs.in);

	std::cout << "dims:" << vox.dims << std::endl;
	std::cout << "res:" << vox.res << std::endl;
	std::cout << "grid2world:" << vox.grid2world << std::endl;


	return 0;
}
