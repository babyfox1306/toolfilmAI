#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace py = pybind11;

// Forward declaration of CUDA functions
#ifdef WITH_CUDA
extern std::pair<cv::Mat, cv::Mat> calculateOpticalFlowCuda(const cv::Mat& prev, const cv::Mat& curr);
extern void initCudaDevice(int device_id);
extern bool checkCudaAvailable();
#endif

// CPU implementation of optical flow using OpenCV
std::pair<py::array_t<float>, py::array_t<float>> calculateOpticalFlowCpu(
    py::array_t<uint8_t> prev_frame, py::array_t<uint8_t> curr_frame, 
    float pyr_scale = 0.5, int levels = 3, int winsize = 15,
    int iterations = 3, int poly_n = 5, float poly_sigma = 1.2) {
    
    // Get dimensions and data pointers from numpy arrays
    auto prev_buffer = prev_frame.request();
    auto curr_buffer = curr_frame.request();
    
    if (prev_buffer.ndim != 2 || curr_buffer.ndim != 2) {
        throw std::runtime_error("Input frames must be grayscale (2D arrays)");
    }
    
    int height = static_cast<int>(prev_buffer.shape[0]);
    int width = static_cast<int>(prev_buffer.shape[1]);
    
    // Convert to OpenCV Mat
    cv::Mat prev(height, width, CV_8UC1, prev_buffer.ptr);
    cv::Mat curr(height, width, CV_8UC1, curr_buffer.ptr);
    
    // Calculate optical flow using Farneback
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(prev, curr, flow, pyr_scale, levels, 
                               winsize, iterations, poly_n, poly_sigma, 0);
    
    // Extract x and y flow components
    std::vector<cv::Mat> flow_components;
    cv::split(flow, flow_components);
    
    // Convert to numpy arrays
    auto flow_x = py::array_t<float>({height, width});
    auto flow_y = py::array_t<float>({height, width});
    
    auto flow_x_ptr = static_cast<float*>(flow_x.request().ptr);
    auto flow_y_ptr = static_cast<float*>(flow_y.request().ptr);
    
    std::memcpy(flow_x_ptr, flow_components[0].data, height * width * sizeof(float));
    std::memcpy(flow_y_ptr, flow_components[1].data, height * width * sizeof(float));
    
    return {flow_x, flow_y};
}

// Wrapper function that chooses between CPU and CUDA implementation
std::pair<py::array_t<float>, py::array_t<float>> calculateOpticalFlow(
    py::array_t<uint8_t> prev_frame, py::array_t<uint8_t> curr_frame,
    bool use_cuda = true, int device_id = 0) {
    
#ifdef WITH_CUDA
    if (use_cuda && checkCudaAvailable()) {
        try {
            // Get dimensions and data pointers from numpy arrays
            auto prev_buffer = prev_frame.request();
            auto curr_buffer = curr_frame.request();
            
            if (prev_buffer.ndim != 2 || curr_buffer.ndim != 2) {
                throw std::runtime_error("Input frames must be grayscale (2D arrays)");
            }
            
            int height = static_cast<int>(prev_buffer.shape[0]);
            int width = static_cast<int>(prev_buffer.shape[1]);
            
            // Convert to OpenCV Mat
            cv::Mat prev(height, width, CV_8UC1, prev_buffer.ptr);
            cv::Mat curr(height, width, CV_8UC1, curr_buffer.ptr);
            
            // Initialize CUDA device
            initCudaDevice(device_id);
            
            // Calculate optical flow with CUDA
            auto [flow_x_mat, flow_y_mat] = calculateOpticalFlowCuda(prev, curr);
            
            // Convert to numpy arrays
            auto flow_x = py::array_t<float>({height, width});
            auto flow_y = py::array_t<float>({height, width});
            
            auto flow_x_ptr = static_cast<float*>(flow_x.request().ptr);
            auto flow_y_ptr = static_cast<float*>(flow_y.request().ptr);
            
            std::memcpy(flow_x_ptr, flow_x_mat.data, height * width * sizeof(float));
            std::memcpy(flow_y_ptr, flow_y_mat.data, height * width * sizeof(float));
            
            return {flow_x, flow_y};
        } catch (const std::exception& e) {
            std::cerr << "CUDA optical flow failed: " << e.what() << ", falling back to CPU" << std::endl;
        }
    }
#endif
    
    // Fall back to CPU implementation
    return calculateOpticalFlowCpu(prev_frame, curr_frame);
}

bool isCudaAvailable() {
#ifdef WITH_CUDA
    return checkCudaAvailable();
#else
    return false;
#endif
}

PYBIND11_MODULE(optical_flow_cpp, m) {
    m.doc() = "Optimized optical flow implementation with CPU and CUDA support";
    
    m.def("calculate_flow", &calculateOpticalFlow, 
          "Calculate optical flow between two frames",
          py::arg("prev_frame"), py::arg("curr_frame"), 
          py::arg("use_cuda") = true, py::arg("device_id") = 0);
    
    m.def("is_cuda_available", &isCudaAvailable, 
          "Check if CUDA support is available");
}
