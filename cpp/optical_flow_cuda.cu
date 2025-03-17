#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cuda/arithm.hpp>
#include <iostream>

// Flag to track CUDA initialization
static bool cuda_initialized = false;

// Check if CUDA is available
bool checkCudaAvailable() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

// Initialize CUDA device
void initCudaDevice(int device_id) {
    if (cuda_initialized) return;
    
    if (checkCudaAvailable()) {
        cudaError_t error = cudaSetDevice(device_id);
        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to set CUDA device: " + 
                                    std::string(cudaGetErrorString(error)));
        }
        cuda_initialized = true;
    } else {
        throw std::runtime_error("No CUDA device available");
    }
}

// Calculate optical flow using CUDA
std::pair<cv::Mat, cv::Mat> calculateOpticalFlowCuda(const cv::Mat& prev, const cv::Mat& curr) {
    try {
        // Upload images to GPU
        cv::cuda::GpuMat d_prev, d_curr;
        d_prev.upload(prev);
        d_curr.upload(curr);
        
        // Create the optical flow object
        cv::Ptr<cv::cuda::FarnebackOpticalFlow> farneback = 
            cv::cuda::FarnebackOpticalFlow::create(
                3,      // num_levels
                0.5,    // pyr_scale  
                false,  // fast_pyramids
                15,     // win_size
                3,      // num_iters
                5,      // poly_n
                1.2,    // poly_sigma
                0       // flags
            );
            
        // Calculate flow
        cv::cuda::GpuMat d_flow;
        farneback->calc(d_prev, d_curr, d_flow);
        
        // Download flow from GPU
        cv::Mat flow;
        d_flow.download(flow);
        
        // Split into x and y components
        cv::Mat flow_x, flow_y;
        std::vector<cv::Mat> flow_parts;
        cv::split(flow, flow_parts);
        
        flow_x = flow_parts[0];
        flow_y = flow_parts[1];
        
        return {flow_x, flow_y};
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV CUDA error: " << e.what() << std::endl;
        throw std::runtime_error("CUDA optical flow calculation failed");
    }
}
