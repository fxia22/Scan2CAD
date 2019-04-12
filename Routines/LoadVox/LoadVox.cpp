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
    py::array_t<float> result(vox.sdf.size());

    py::buffer_info buf = result.request();
    float * ptr = (float *) buf.ptr;

    for (size_t idx = 0; idx < vox.sdf.size(); idx++) ptr[idx] = vox.sdf[idx];

    return result;

}


py::array_t<float> load_vox_with_pdf_np(std::string filename) {

    Vox vox = load_vox(filename);
    py::array_t<float> result(vox.sdf.size() + vox.pdf.size());

    py::buffer_info buf = result.request();
    float * ptr = (float *) buf.ptr;

    for (size_t idx = 0; idx < vox.sdf.size(); idx++) ptr[idx] = vox.sdf[idx];
    for (size_t idx = 0; idx < vox.pdf.size(); idx++) ptr[idx + vox.sdf.size()] = vox.pdf[idx];

    return result;
}


PYBIND11_MODULE(LoadVox, m) {
    m.def("load_vox_np", &load_vox_np, "load vox to numpy function")
     .def("load_vox_with_pdf_np", &load_vox_with_pdf_np, "load vox to numpy function");
}
