#include <cmath>
#include <fstream>
#include <random>
#include <chrono>
#include <unordered_map>
#include <iostream>

#include <eigen3/Eigen/Dense>

#include "LoaderVOX.h"
#include "LoaderMesh.h"

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


py::array_t<float> load_vox_np(std::string filename) {
   
    Vox vox = load_vox(filename);
    std::cout << vox.dims << std::endl;
    py::array_t<double> result(vox.sdf.size());

    return result;

}

PYBIND11_MODULE(LoadVox, m) {
    m.def("load_vox_np", &load_vox_np, "load vox to numpy function");
}
