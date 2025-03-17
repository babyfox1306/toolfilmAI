#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <thread>

namespace py = pybind11;

// Structure to hold video information
struct VideoInfo {
    int width;
    int height;
    double fps;
    int frame_count;
    double duration;
};

// Get basic video information
VideoInfo getVideoInfo(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open video: " + video_path);
    }
    
    VideoInfo info;
    info.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    info.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    info.fps = cap.get(cv::CAP_PROP_FPS);
    info.frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    info.duration = info.frame_count / info.fps;
    
    cap.release();
    return info;
}

// Extract frames at specified indices
py::tuple extractFramesAtIndices(const std::string& video_path, 
                                const std::vector<int>& frame_indices,
                                int resize_width = 0, int resize_height = 0) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open video: " + video_path);
    }
    
    int frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    // Validate frame indices
    for (int idx : frame_indices) {
        if (idx < 0 || idx >= frame_count) {
            throw std::runtime_error("Frame index out of range: " + std::to_string(idx));
        }
    }
    
    // Extract frames at specified indices
    std::vector<py::array_t<uint8_t>> frames;
    std::vector<double> timestamps;
    
    cv::Mat frame;
    for (int idx : frame_indices) {
        cap.set(cv::CAP_PROP_POS_FRAMES, idx);
        bool success = cap.read(frame);
        
        if (!success) {
            continue;  // Skip failed frames
        }
        
        // Resize if requested
        if (resize_width > 0 && resize_height > 0) {
            cv::resize(frame, frame, cv::Size(resize_width, resize_height));
        }
        
        // Convert to RGB if necessary
        cv::Mat rgb_frame;
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
        
        // Create numpy array from frame data
        auto height = rgb_frame.rows;
        auto width = rgb_frame.cols;
        auto channels = rgb_frame.channels();
        
        py::array_t<uint8_t> np_frame({height, width, channels});
        std::memcpy(np_frame.mutable_data(), rgb_frame.data, height * width * channels);
        
        frames.push_back(np_frame);
        timestamps.push_back(idx / fps);
    }
    
    cap.release();
    
    // Convert to NumPy arrays
    py::list frame_list;
    for (const auto& frame : frames) {
        frame_list.append(frame);
    }
    
    py::array_t<double> timestamps_array(timestamps.size());
    std::memcpy(timestamps_array.mutable_data(), timestamps.data(), timestamps.size() * sizeof(double));
    
    return py::make_tuple(frame_list, timestamps_array, fps);
}

// Extract frames at specified rate (e.g., 1 frame per second)
py::tuple extractFramesWithSampleRate(const std::string& video_path, 
                                    double sample_rate = 1.0,
                                    int resize_width = 0, int resize_height = 0,
                                    int max_frames = -1) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open video: " + video_path);
    }
    
    int frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    // Calculate frame indices based on sample rate
    int frame_interval = static_cast<int>(fps / sample_rate);
    if (frame_interval < 1) frame_interval = 1;
    
    std::vector<int> frame_indices;
    for (int i = 0; i < frame_count; i += frame_interval) {
        frame_indices.push_back(i);
        if (max_frames > 0 && frame_indices.size() >= static_cast<size_t>(max_frames)) {
            break;
        }
    }
    
    cap.release();
    
    // Extract frames at calculated indices
    return extractFramesAtIndices(video_path, frame_indices, resize_width, resize_height);
}

PYBIND11_MODULE(video_extract_cpp, m) {
    m.doc() = "Optimized video frame extraction";
    
    m.def("get_video_info", &getVideoInfo, 
          "Get basic information about a video file",
          py::arg("video_path"));
    
    m.def("extract_frames_at_indices", &extractFramesAtIndices, 
          "Extract frames at specified indices",
          py::arg("video_path"), py::arg("frame_indices"),
          py::arg("resize_width") = 0, py::arg("resize_height") = 0);
    
    m.def("extract_frames_with_sample_rate", &extractFramesWithSampleRate,
          "Extract frames at specified sample rate",
          py::arg("video_path"), py::arg("sample_rate") = 1.0,
          py::arg("resize_width") = 0, py::arg("resize_height") = 0,
          py::arg("max_frames") = -1);
          
    // Register VideoInfo class
    py::class_<VideoInfo>(m, "VideoInfo")
        .def_readonly("width", &VideoInfo::width)
        .def_readonly("height", &VideoInfo::height)
        .def_readonly("fps", &VideoInfo::fps)
        .def_readonly("frame_count", &VideoInfo::frame_count)
        .def_readonly("duration", &VideoInfo::duration);
}
