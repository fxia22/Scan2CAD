#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include <unordered_map>
#include <iostream>

#include <eigen3/Eigen/Dense>
#include "LoaderVOX.h"
#include "LoaderMesh.h"
#include "SE3.h"
#include "Colormap.h"
#include "args.hxx"

struct InputArgs {
	std::string in;
	std::string out;
	bool is_unitless;
	bool redcenter;
	std::string cmap;
	float trunc;
} inargs;


struct Vox {
	Eigen::Vector3i grid_dims;
	Eigen::Matrix4f grid2world;
	float res;
	std::vector<float> sdf;
	std::vector<float> pdf;
};


void parse_args(int argc, char** argv) {
	args::ArgumentParser parser("This is a test program.", "This goes after the options.");
	args::Group allgroup(parser, "", args::Group::Validators::All);

	args::ValueFlag<std::string> in(allgroup, "bunny.sdf", "vox file", {"in"});
	args::ValueFlag<std::string> out(allgroup, "bunny.vox", "out file", {"out"});
	args::ValueFlag<bool> is_unitless(parser, "false", "normalize voxel grid or no units?", {"is_unitless"}, false);
	args::ValueFlag<bool> redcenter(parser, "false", "red center in grid?", {"redcenter"}, false);
	args::ValueFlag<std::string> cmap(parser, "jet, inferno, magma, viridis, gray2red, beige2red", "color map format", {"cmap"}, "jet");
	args::ValueFlag<float> trunc(parser, "1.0", "truncation for visible voxels", {"trunc"}, 1.0);


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
	inargs.out = args::get(out);
	inargs.cmap = args::get(cmap);
	inargs.redcenter = args::get(redcenter);
	inargs.trunc = args::get(trunc);
	inargs.is_unitless = args::get(is_unitless);

};

int main(int argc, char** argv) {
   	parse_args(argc, argv); 

	Vox vox;
	Eigen::Vector3f voxelsize(1, 1, 1);

	std::cout << inargs.in << " " << inargs.out << std::endl;
	std::ifstream inFile(inargs.in);
	
	Eigen::Vector3f offset;

	if (inFile.is_open()) {
		inFile >> vox.grid_dims[0] >> vox.grid_dims[1] >> vox.grid_dims[2];
		std::cout << vox.grid_dims[0] << " " << vox.grid_dims[1] << " " << vox.grid_dims[2] << std::endl;

		inFile >> offset[0] >> offset[1] >> offset[2];
		inFile >> vox.res;
		std::cout << vox.res << std::endl;

		vox.grid2world << 1, 0, 0, offset[0],
						  0, 1, 0, offset[1],
						  0, 0, 1, offset[2],
						  0, 0, 0, 1;

		std::cout << vox.grid2world << std::endl;

		int n_elems = vox.grid_dims(0)*vox.grid_dims(1)*vox.grid_dims(2);	

		float data;
		while (inFile >> data) {
			vox.sdf.push_back(data);
			std::cout << data << std::endl;
		}
	}

	save_vox<float, 1, float, 1> (inargs.out, vox.grid_dims, vox.res, vox.grid2world, vox.sdf, vox.pdf);


	/*if (inargs.in.find(".vox2") != std::string::npos)
		load_vox<float, 1, float, 1>(inargs.in, vox.grid_dims, vox.res, vox.grid2world, vox.sdf, vox.pdf);
	else if (inargs.in.find(".vox") != std::string::npos)
		load_vox<float, 1>(inargs.in, vox.grid_dims, vox.res, vox.grid2world, vox.sdf);
	else if (inargs.in.find(".df") != std::string::npos) {
		load_vox<float, 1>(inargs.in, vox.grid_dims, vox.res, vox.grid2world, vox.sdf, false);	
	} else {
		fprintf(stderr, "Error: Grid format not known.\n");
		std::exit(1);
	}*/



	/*if (inargs.is_unitless) {
		Eigen::Vector3f t;
		Eigen::Quaternionf q;
		Eigen::Vector3f s;
		decompose_mat4(vox.grid2world, t, q, s);
		voxelsize = s;
	}*/

	/*
	if (vox.pdf.size() == 0) {
		vox.pdf.resize(vox.sdf.size());
		std::fill(vox.pdf.begin(), vox.pdf.end(), 0);
		if (inargs.redcenter) {
			int c = vox.grid_dims(0)/2;
			int dim = vox.grid_dims(0);
			int w = 1;
			for (int i = c - w; i < c + w + 1; i++)
				for (int j = c - w; j < c + w + 1; j++)
					for (int k = c - w; k < c + w + 1; k++)
						vox.pdf[i*dim*dim + j*dim + k] = 1;
		}
	}*/


	return 0;
}
