#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

// Include FFmpeg headers
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

namespace py = pybind11;

class FFmpegError : public std::runtime_error {
public:
    FFmpegError(const std::string& msg) : std::runtime_error(msg) {}
};

// Helper function for FFmpeg error handling
static void check_ffmpeg_error(int err, const std::string& action) {
    if (err < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(err, errbuf, AV_ERROR_MAX_STRING_SIZE);
        std::string error_msg = action + " failed: " + std::string(errbuf);
        throw FFmpegError(error_msg);
    }
}

// Concatenate video segments directly using FFmpeg API
bool concatenateVideos(const std::vector<std::string>& input_files, 
                      const std::string& output_file,
                      const std::string& codec = "libx264") {
    try {
        // Initialize FFmpeg
        #if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
            av_register_all();
        #endif
        avformat_network_init();
        
        // Check if input files exist
        if (input_files.empty()) {
            throw FFmpegError("No input files provided");
        }
        
        // Create output context
        AVFormatContext *output_ctx = nullptr;
        int ret = avformat_alloc_output_context2(&output_ctx, nullptr, nullptr, output_file.c_str());
        check_ffmpeg_error(ret, "Creating output context");
        
        // Open the output file
        ret = avio_open(&output_ctx->pb, output_file.c_str(), AVIO_FLAG_WRITE);
        check_ffmpeg_error(ret, "Opening output file");
        
        // Get info from the first input file to set up streams in output
        AVFormatContext *input_ctx = nullptr;
        ret = avformat_open_input(&input_ctx, input_files[0].c_str(), nullptr, nullptr);
        check_ffmpeg_error(ret, "Opening input file " + input_files[0]);
        
        ret = avformat_find_stream_info(input_ctx, nullptr);
        check_ffmpeg_error(ret, "Finding stream info");
        
        // Map streams from input to output
        std::vector<int> stream_mapping;
        for (unsigned int i = 0; i < input_ctx->nb_streams; i++) {
            AVStream *in_stream = input_ctx->streams[i];
            AVStream *out_stream = avformat_new_stream(output_ctx, nullptr);
            if (!out_stream) {
                throw FFmpegError("Failed to create output stream");
            }
            
            // Copy codec parameters
            ret = avcodec_parameters_copy(out_stream->codecpar, in_stream->codecpar);
            check_ffmpeg_error(ret, "Copying codec parameters");
            
            // Ensure codec tag is compatible
            out_stream->codecpar->codec_tag = 0;
            
            // Add to mapping
            stream_mapping.push_back(i);
        }
        
        // Write header
        ret = avformat_write_header(output_ctx, nullptr);
        check_ffmpeg_error(ret, "Writing header");
        
        // Process each input file
        AVPacket packet;
        av_init_packet(&packet);
        packet.data = nullptr;
        packet.size = 0;
        
        int64_t pts_offset = 0;
        int64_t dts_offset = 0;
        
        for (const auto& input_file : input_files) {
            // Open input file
            if (input_ctx) {
                avformat_close_input(&input_ctx);
            }
            
            ret = avformat_open_input(&input_ctx, input_file.c_str(), nullptr, nullptr);
            check_ffmpeg_error(ret, "Opening input file " + input_file);
            
            ret = avformat_find_stream_info(input_ctx, nullptr);
            check_ffmpeg_error(ret, "Finding stream info");
            
            // Copy packets
            while (av_read_frame(input_ctx, &packet) >= 0) {
                if (packet.stream_index < static_cast<int>(stream_mapping.size())) {
                    // Get stream mapping
                    int out_stream_index = stream_mapping[packet.stream_index];
                    
                    // Adjust packet timestamps
                    AVStream *in_stream = input_ctx->streams[packet.stream_index];
                    AVStream *out_stream = output_ctx->streams[out_stream_index];
                    
                    // Rescale timestamps
                    if (packet.pts != AV_NOPTS_VALUE) {
                        packet.pts = av_rescale_q_rnd(packet.pts, in_stream->time_base, out_stream->time_base, 
                                                     static_cast<AVRounding>(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
                        packet.pts += pts_offset;
                    }
                    
                    if (packet.dts != AV_NOPTS_VALUE) {
                        packet.dts = av_rescale_q_rnd(packet.dts, in_stream->time_base, out_stream->time_base,
                                                     static_cast<AVRounding>(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
                        packet.dts += dts_offset;
                    }
                    
                    // Set stream index to output
                    packet.stream_index = out_stream_index;
                    
                    // Write packet
                    ret = av_interleaved_write_frame(output_ctx, &packet);
                    check_ffmpeg_error(ret, "Writing frame");
                }
                
                av_packet_unref(&packet);
            }
            
            // Update timestamp offsets for next file
            for (unsigned int i = 0; i < input_ctx->nb_streams; i++) {
                AVStream *in_stream = input_ctx->streams[i];
                if (i < stream_mapping.size()) {
                    int out_stream_index = stream_mapping[i];
                    AVStream *out_stream = output_ctx->streams[out_stream_index];
                    
                    // Get the duration in output timebase
                    int64_t duration = av_rescale_q(in_stream->duration, in_stream->time_base, out_stream->time_base);
                    pts_offset += duration;
                    dts_offset += duration;
                }
            }
        }
        
        // Write trailer
        ret = av_write_trailer(output_ctx);
        check_ffmpeg_error(ret, "Writing trailer");
        
        // Cleanup
        if (input_ctx) {
            avformat_close_input(&input_ctx);
        }
        
        avio_closep(&output_ctx->pb);
        avformat_free_context(output_ctx);
        
        return true;
        
    } catch (const FFmpegError& e) {
        std::cerr << "FFmpeg error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

PYBIND11_MODULE(video_concat_cpp, m) {
    m.doc() = "Direct FFmpeg API integration for video concatenation";
    
    m.def("concatenate_videos", &concatenateVideos, 
          "Concatenate multiple video files using FFmpeg API",
          py::arg("input_files"), py::arg("output_file"),
          py::arg("codec") = "libx264");
}
