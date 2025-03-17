import os
import torch
import argparse
import cv2
import numpy as np
from video_processing import extract_important_clips
from audio_processing import extract_audio, filter_audio, initialize_whisper_model, transcribe_audio_optimized, generate_voice_over
# Add missing imports for the undefined functions
from audio_processing import transcribe_audio_faster, transcribe_audio_parallel
from action_detection import detect_climax_scenes, extract_climax_scenes, fast_motion_detection, fast_extract_segments
from summary_generator import summarize_transcript, summarize_transcript_with_textrank, enhanced_summarize_transcript, filter_irrelevant_content, clean_transcript, validate_summary
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import json
import signal
import subprocess
import shutil
import time
import sys
import threading
from utils.gpu_utils import force_gpu_usage, build_cpp_extensions

def setup_directories():
    """Tạo các thư mục cần thiết"""
    os.makedirs('input_videos', exist_ok=True)
    os.makedirs('output_videos', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def setup_device(use_cuda=True):
    """Thiết lập và kiểm tra device (CPU/GPU)"""
    # THÊM: Ưu tiên biến môi trường FORCE_CUDA
    force_cuda = os.environ.get('FORCE_CUDA', '0') == '1'
    
    has_cuda = torch.cuda.is_available()
    if (has_cuda or force_cuda):
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Forced CUDA Device"
        print(f"🔍 Phát hiện GPU: {device_name}")
        if not use_cuda:
            print("⚠️ GPU bị tắt theo tham số --use-cuda=False")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.cuda.empty_cache()  # Xóa cache CUDA
                print(f"🚀 GPU đã được kích hoạt: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB RAM")
            else:
                print("🚀 GPU được kích hoạt qua chế độ FORCE_CUDA")
    else:
        device = torch.device('cpu')
        print("⚠️ Không phát hiện GPU, sẽ sử dụng CPU")

    return device, has_cuda or force_cuda  # Trả về has_cuda hoặc force_cuda

def optimize_cuda_memory():
    """Tối ưu bộ nhớ CUDA để đạt hiệu suất tốt nhất"""
    if torch.cuda.is_available():
        # Xóa cache hiện tại
        torch.cuda.empty_cache()
        
        # Áp dụng các tối ưu cho PyTorch CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
        
        # Quản lý bộ nhớ thông minh
        total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory total: {total_gpu_mem:.1f} GB")
        
        if total_gpu_mem > 6.0:
            print("❇️ GPU lớn, kích hoạt tối ưu memory cho video nặng")
            if hasattr(torch.cuda, 'memory_stats'):
                torch.cuda.memory_stats()
        else:
            print("⚠️ GPU nhỏ, áp dụng tối ưu cho thiết bị hạn chế")
            
        return True
    return False

def load_yolo_model(device):
    """Tải model YOLO"""
    # Kiểm tra & tải YOLOv5 nếu chưa có
    model_path = os.path.join('models', 'yolov5s.pt')
    if not os.path.exists(model_path):
        print(f"⚠️ Không tìm thấy model tại {model_path}")
        # Thử kiểm tra trong thư mục gốc
        alt_model_path = 'yolov5s.pt'
        if (os.path.exists(alt_model_path)):
            print(f"✅ Đã tìm thấy model tại {alt_model_path}")
            model_path = alt_model_path
        else:
            print("Đang tải model YOLOv5...")
            os.makedirs('models', exist_ok=True)
            try:
                torch.hub.download_url_to_file(
                    'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt', 
                    model_path
                )
            except Exception as e:
                print(f"⛔ Không thể tải model từ GitHub: {e}")
                print("Vui lòng tải thủ công model YOLOv5s từ:")
                print("https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt")
                print(f"và đặt vào thư mục {os.path.dirname(model_path)} với tên 'yolov5s.pt'")
                exit(1)

    # Load YOLOv5 model
    print(f"Đang khởi tạo model YOLOv5 từ {model_path}...")
    try:
        # Thử tải với torch.hub
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False, trust_repo=True)
    except Exception as e:
        print(f"Lỗi khi tải model qua torch.hub: {e}")
        print("Thử tải model trực tiếp...")
        
        try:
            # Tải trực tiếp với PyTorch
            model = torch.load(model_path, map_location=device)
            if hasattr(model, 'module'):  # Xử lý trường hợp model được lưu với DataParallel
                model = model.module
        except Exception as e2:
            print(f"⛔ Không thể tải model: {e2}")
            exit(1)

    model.to(device)  # Chuyển mô hình lên GPU nếu có
    model.conf = 0.25  # Ngưỡng tin cậy
    model.iou = 0.45   # Ngưỡng IoU
    if device.type == 'cuda':
        model.amp = True  # Sử dụng mixed precision khi có GPU
        print("✅ YOLO model được tải lên GPU với Automatic Mixed Precision (AMP)")
    
    return model

def has_dialogue(audio_path, threshold_db=-35, min_dialogue_duration=0.5):
    """Kiểm tra nhanh xem video có lời thoại hay không"""
    try:
        from pydub import AudioSegment
        import numpy as np
        
        print("Đang kiểm tra nhanh lời thoại...")
        audio = AudioSegment.from_file(audio_path)
        
        # Lấy mẫu audio
        samples = np.array(audio.get_array_of_samples())
        if (audio.channels == 2):
            samples = samples.reshape((-1, 2)).mean(axis=1)  # Chuyển stereo thành mono
            
        # Chuyển sang dB
        samples = np.abs(samples)
        max_amplitude = np.iinfo(samples.dtype).max
        samples_db = 20 * np.log10(samples / max_amplitude + 1e-10)
        
        # Kiểm tra các đoạn có âm thanh trên ngưỡng
        samples_above_threshold = samples_db > threshold_db
        
        # Tính số lượng mẫu liên tiếp trên ngưỡng
        count = 0
        max_count = 0
        for is_above in samples_above_threshold:
            if is_above:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
                
        # Kiểm tra có đủ mẫu liên tiếp không
        min_samples = min_dialogue_duration * audio.frame_rate
        has_speech = max_count >= min_samples
        
        if has_speech:
            print("✅ Phát hiện lời thoại trong video")
        else:
            print("ℹ️ Không phát hiện lời thoại rõ ràng trong video")
            
        return has_speech
        
    except Exception as e:
        print(f"⚠️ Lỗi khi kiểm tra lời thoại: {e}")
        # Nếu có lỗi, giả định là có lời thoại để an toàn
        return True

def process_audio_task(input_path, device, ratio=0.3):
    """Tác vụ xử lý audio chạy trong process riêng"""
    try:
        print("\n🔊 Đang xử lý âm thanh và nhận dạng lời thoại...")
        # Khởi tạo model Whisper
        whisper_model = initialize_whisper_model(device)
        
        # Trích xuất và xử lý âm thanh
        raw_audio_path = os.path.join('output_videos', 'temp_audio_raw.wav')
        filtered_audio_path = os.path.join('output_videos', 'temp_audio.wav')
        extract_audio(input_path, raw_audio_path)
        
        # Kiểm tra nhanh xem video có lời thoại không
        if not has_dialogue(raw_audio_path):
            print("⚠️ Video không có lời thoại rõ ràng, không cần xử lý tiếp audio")
            result = {
                "success": False,
                "reason": "no_dialogue",
                "transcript": "",
                "summary": "Video không có lời thoại."
            }
            return result
            
        filter_audio(raw_audio_path, filtered_audio_path)
        
        # Nhận dạng giọng nói
        transcript = transcribe_audio_optimized(filtered_audio_path, whisper_model)
        transcript = clean_transcript(transcript)
        
        if not transcript or len(transcript.split()) < 5:
            print("⚠️ Không nhận diện được lời thoại có nghĩa")
            result = {
                "success": False,
                "reason": "empty_transcript",
                "transcript": transcript,
                "summary": "Không nhận diện được lời thoại có nghĩa."
            }
            return result
            
        # Lọc và tóm tắt nội dung
        filtered_transcript = filter_irrelevant_content(transcript)
        summary = enhanced_summarize_transcript(filtered_transcript, ratio)
        
        # Lưu transcript và summary
        full_transcript_path = os.path.join('output_videos', 'review_transcript.txt')
        summary_path = os.path.join('output_videos', 'review_summary.txt')
        
        with open(full_transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        
        print(f"✅ Đã lưu bản tóm tắt review phim vào: {summary_path}")
        
        # Tạo voice-over cho bản tóm tắt
        voice_over_path = os.path.join('output_videos', 'review_voiceover.mp3')
        generate_voice_over(summary, voice_over_path)
        
        # Xóa file âm thanh tạm
        if os.path.exists(raw_audio_path):
            os.remove(raw_audio_path)
        if os.path.exists(filtered_audio_path):
            os.remove(filtered_audio_path)
        
        result = {
            "success": True,
            "transcript": transcript,
            "summary": summary,
            "full_transcript_path": full_transcript_path,
            "summary_path": summary_path,
            "voice_over_path": voice_over_path
        }
        return result
        
    except Exception as e:
        print(f"❌ Lỗi khi xử lý audio: {e}")
        result = {
            "success": False,
            "reason": "error",
            "error": str(e)
        }
        return result

def process_video_task(input_path, output_path, model, device):
    """Tác vụ xử lý video chạy trong thread riêng"""
    try:
        print("\n🎬 Đang phân tích video để tìm cảnh quan trọng...")
        
        # Tùy chọn phương pháp phát hiện - mặc định sử dụng phương pháp kết hợp
        detection_method = 3
        buffer_time = 1.0
        
        # Thực hiện phát hiện theo phương pháp đã chọn
        if detection_method == 1:
            # Phát hiện người xuất hiện
            success = extract_important_clips(input_path, output_path, model, device, buffer_time, sample_rate=1)
        elif detection_method == 2:
            # Phát hiện cảnh cao trào
            success = extract_climax_scenes(input_path, output_path, buffer_time)
        else:
            # Kết hợp cả hai phương pháp
            from action_detection import extract_climax_scenes_improved
            success = extract_climax_scenes_improved(input_path, output_path, model, device, buffer_time)
        
        result = {
            "success": success,
            "output_path": output_path
        }
        return result
        
    except Exception as e:
        print(f"❌ Lỗi khi xử lý video: {e}")
        result = {
            "success": False,
            "reason": "error",
            "error": str(e)
        }
        return result

def process_both_modes_parallel(input_path, output_dir, model, device, summary_ratio=0.3):
    """Xử lý cả hai chế độ song song"""
    print("\n=== ĐANG XỬ LÝ SONG SONG CHẾ ĐỘ REVIEW VÀ SHORT ===")
    
    # Đường dẫn output
    short_output_path = os.path.join(output_dir, 'short.mp4')
    
    # Tạo pool cho các process và thread
    audio_process = None
    video_future = None
    
    # Chuẩn bị device cho từng process
    if device.type == 'cuda' and torch.cuda.device_count() >= 2:
        # Nếu có 2 GPU trở lên, mỗi process dùng 1 GPU
        audio_device = torch.device('cuda:0')
        video_device = torch.device('cuda:1')
        print("🚀 Phát hiện nhiều GPU, phân bổ tác vụ trên các GPU khác nhau")
    else:
        # Nếu chỉ có 1 GPU hoặc CPU, dùng chung
        audio_device = device
        video_device = device
    
    try:
        # Khởi động quy trình xử lý audio trong một process riêng
        audio_process = multiprocessing.Process(
            target=process_audio_wrapper,
            args=(input_path, audio_device, summary_ratio, output_dir)
        )
        audio_process.start()
        
        # Khởi động quy trình xử lý video trong thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            video_future = executor.submit(
                process_video_task, input_path, short_output_path, model, video_device
            )
        
        # Đợi và lấy kết quả từ thread xử lý video
        video_result = video_future.result()
        
        # In thông báo đang đợi xử lý audio
        print("\n⏳ Đang đợi xử lý audio hoàn tất...")
        
        # Đợi process audio hoàn tất
        audio_process.join(timeout=300)  # Timeout 5 phút
        
        # Kiểm tra xem cả hai quá trình đã hoàn tất chưa
        if not audio_process.is_alive() and video_result["success"]:
            # Đọc kết quả từ file tạm
            audio_result_path = os.path.join(output_dir, 'audio_result.json')
            if os.path.exists(audio_result_path):
                import json
                with open(audio_result_path, 'r', encoding='utf-8') as f:
                    audio_result = json.load(f)
                
                # Xóa file tạm
                os.remove(audio_result_path)
            else:
                audio_result = {"success": False}
                
            # Tạo video review với voiceover nếu cả hai quá trình thành công
            if audio_result["success"] and video_result["success"]:
                voice_over_path = audio_result.get("voice_over_path")
                short_video_path = video_result.get("output_path")
                
                if os.path.exists(voice_over_path) and os.path.exists(short_video_path):
                    create_video_with_voiceover(short_video_path, voice_over_path, output_dir)
        
        else:
            print("⚠️ Một hoặc cả hai quá trình xử lý không hoàn tất đúng hạn.")
            
            # Ngắt process nếu vẫn đang chạy
            if audio_process.is_alive():
                print("⛔ Đang dừng quá trình xử lý audio...")
                audio_process.terminate()
                
    except Exception as e:
        print(f"❌ Lỗi khi xử lý song song: {e}")
        
        # Dừng các process đang chạy

        if audio_process and audio_process.is_alive():
            audio_process.terminate()
            
    finally:
        # Dọn dẹp tài nguyên
        audio_result_path = os.path.join(output_dir, 'audio_result.json')
        if os.path.exists(audio_result_path):
            os.remove(audio_result_path)
            
def process_audio_wrapper(input_path, device, ratio, output_dir):
    """Wrapper để chạy process_audio_task trong process riêng và lưu kết quả"""
    try:
        # Chuyển đổi torch device thành string để truyền qua process
        device_str = 'cuda' if device.type == 'cuda' else 'cpu'
        device = torch.device(device_str)
        
        # Thực hiện xử lý audio
        result = process_audio_task(input_path, device, ratio)
        
        # Lưu kết quả vào file để process chính có thể đọc
        import json
        audio_result_path = os.path.join(output_dir, 'audio_result.json')
        with open(audio_result_path, 'w', encoding='utf-8') as f:
            # Chuyển đổi đường dẫn thành string nếu cần
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (str, bool, int, float)):
                    serializable_result[key] = value
                elif value is None:
                    serializable_result[key] = None
                else:
                    serializable_result[key] = str(value)
            json.dump(serializable_result, f)
            
    except Exception as e:
        print(f"❌ Lỗi trong process audio: {e}")
        # Lưu thông tin lỗi
        import json
        audio_result_path = os.path.join(output_dir, 'audio_result.json')
        with open(audio_result_path, 'w', encoding='utf-8') as f:
            json.dump({"success": False, "error": str(e)}, f)

def create_video_with_voiceover(video_path, audio_path, output_dir):
    """Tạo video review với voiceover"""
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
        
        print("\n=== TẠO VIDEO REVIEW WITH VOICEOVER ===")
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        
        # Lấy độ dài ngắn hơn giữa video và audio để tránh lỗi
        min_duration = min(video_clip.duration, audio_clip.duration)
        video_clip = video_clip.subclip(0, min_duration)
        audio_clip = audio_clip.subclip(0, min_duration)
        
        # Thêm voiceover vào video
        final_clip = video_clip.set_audio(audio_clip)
        review_video_path = os.path.join(output_dir, 'review_with_voiceover.mp4')
        final_clip.write_videofile(review_video_path, codec="libx264", threads=min(4, os.cpu_count()))
        
        print(f"✅ Đã tạo video review kèm lời thoại: {review_video_path}")
        
        # Đóng các clip
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        
    except Exception as e:
        print(f"❌ Lỗi khi tạo video review kèm lời thoại: {e}")

def process_review_mode(input_path, output_path, device, model=None, ratio=0.3, 
                       use_faster_whisper=True, use_cuda=True):
    """Chế độ Review Phim - Tóm tắt nội dung theo lời thoại và cảnh cao trào"""
    print("\n=== CHẾ ĐỘ REVIEW PHIM ===")
    
    # Phân tích cao trào trước để chạy song song với xử lý audio
    print("🎬 Đang phát hiện các cảnh cao trào để bổ sung vào review...")
    
    # Nếu model chưa được truyền vào, tải YOLO model
    if (model is None):
        try:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.to(device)
            if device.type == 'cuda':
                model.amp = True  # Enable mixed precision for better performance
                print("✅ YOLO model sử dụng GPU với Automatic Mixed Precision")
        except Exception as e:
            print(f"⚠️ Không thể tải model YOLO: {e}")
            print("Tiếp tục mà không có phân tích cao trào")
            model = None
    
    # Tạo một thread riêng để phát hiện cao trào
    climax_segments = []
    climax_future = None
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        if model is not None:
            try:
                from action_detection import detect_climax_scenes_improved
                # Bắt đầu tiến trình phát hiện cao trào trong thread riêng
                climax_future = executor.submit(
                    detect_climax_scenes_improved, input_path, model, device, threshold_multiplier=0.9
                )
            except Exception as e:
                print(f"⚠️ Lỗi khi bắt đầu phát hiện cao trào: {e}")
    
    # Sử dụng Faster-Whisper nếu được yêu cầu và có GPU
    transcript = ""
    if use_faster_whisper and torch.cuda.is_available():
        try:
            print("🚀 Đang sử dụng Faster-Whisper để tăng tốc 3-5 lần...")
            
            # Trích xuất và xử lý âm thanh
            raw_audio_path = os.path.join('output_videos', 'temp_audio_raw.wav')
            filtered_audio_path = os.path.join('output_videos', 'temp_audio.wav')
            extract_audio(input_path, raw_audio_path)
            
            # Kiểm tra nhanh xem video có lời thoại không
            has_speech = has_dialogue(raw_audio_path)
            
            if has_speech:
                # Tiếp tục với quy trình phân tích lời thoại thông thường
                filter_audio(raw_audio_path, filtered_audio_path)
                
                # Nhận dạng giọng nói với Faster-Whisper
                print("🔍 Phân tích nội dung lời thoại...")
                result = transcribe_audio_faster(filtered_audio_path)
                if result and result.get("text"):
                    transcript = result["text"] 
                    transcript = clean_transcript(transcript)
            else:
                print("⚠️ Không phát hiện được lời thoại rõ ràng")
                transcript = ""
        except Exception as e:
            print(f"⚠️ Lỗi khi sử dụng Faster-Whisper: {e}")
            print("Chuyển sang phương pháp song song...")
            use_faster_whisper = False
    
    # Thử phương pháp song song nếu Faster-Whisper thất bại
    if not transcript and use_faster_whisper:
        try:
            print("Thử dùng phương pháp song song với các chunk...")
            raw_audio_path = os.path.join('output_videos', 'temp_audio_raw.wav')
            filtered_audio_path = os.path.join('output_videos', 'temp_audio.wav')
            if not os.path.exists(filtered_audio_path):
                extract_audio(input_path, raw_audio_path)
                filter_audio(raw_audio_path, filtered_audio_path)
            
            result = transcribe_audio_parallel(
                filtered_audio_path, 
                chunk_length_sec=15,
                max_workers=4
            )
            if result and result["text"]:
                transcript = result["text"]
                transcript = clean_transcript(transcript)
        except Exception as e:
            print(f"⚠️ Lỗi khi dùng phương pháp song song: {e}")
            print("Chuyển sang Whisper thường...")
    
    # Nếu hai phương pháp trên thất bại, dùng phương pháp thông thường
    if not transcript:
        # Khởi tạo model Whisper
        whisper_model = initialize_whisper_model(device)
        
        # Trích xuất và xử lý âm thanh nếu chưa có
        raw_audio_path = os.path.join('output_videos', 'temp_audio_raw.wav')
        filtered_audio_path = os.path.join('output_videos', 'temp_audio.wav')
        if not os.path.exists(raw_audio_path):
            extract_audio(input_path, raw_audio_path)
        if not os.path.exists(filtered_audio_path):
            filter_audio(raw_audio_path, filtered_audio_path)
        
        # Kiểm tra nhanh xem video có lời thoại không
        has_speech = has_dialogue(raw_audio_path)
        
        if has_speech:
            # Nhận dạng giọng nói
            print("🔍 Phân tích nội dung lời thoại...")
            transcript = transcribe_audio_optimized(filtered_audio_path, whisper_model)
            transcript = clean_transcript(transcript)
        else:
            print("⚠️ Không phát hiện được lời thoại rõ ràng")
            transcript = ""
    
    # Lấy kết quả từ thread phát hiện cao trào (nếu có)
    action_summary = []
    if climax_future:
        try:
            climax_segments = climax_future.result()
            if climax_segments:
                print(f"🎭 Phát hiện được {len(climax_segments)} cảnh cao trào")
                action_summary = [f"Cảnh cao trào từ {start:.1f}s đến {end:.1f}s" for start, end in climax_segments]
            else:
                print("⚠️ Không phát hiện được cảnh cao trào nào")
        except Exception as e:
            print(f"⚠️ Lỗi khi phát hiện cao trào: {e}")
    
    # LOGIC THÔNG MINH: Kết hợp tóm tắt lời thoại và cao trào
    if transcript and len(transcript.split()) >= 5:
        # Lọc và tóm tắt nội dung lời thoại
        filtered_transcript = filter_irrelevant_content(transcript)
        summary = enhanced_summarize_transcript(filtered_transcript, ratio)
        
        # Bổ sung thông tin cao trào vào cuối bản tóm tắt
        if action_summary:
            summary += "\n\n## Các cảnh cao trào:\n"
            summary += "\n".join([f"- {item}" for item in action_summary])
    else:
        # Nếu không có lời thoại đáng kể, tạo tóm tắt từ cảnh cao trào
        if action_summary:
            print("📝 Tạo tóm tắt dựa trên cảnh cao trào")
            summary = "Video không có lời thoại rõ ràng hoặc có rất ít lời thoại.\n\n"
            summary += "## Tóm tắt dựa trên phân tích hình ảnh:\n\n"
            summary += "Video chứa các cảnh cao trào sau:\n"
            summary += "\n".join([f"- {item}" for item in action_summary])
        else:
            # Không có cả lời thoại và cao trào
            summary = "Video không có lời thoại rõ ràng và không phát hiện được cảnh cao trào đáng chú ý."
    
    # Lưu transcript và summary
    full_transcript_path = os.path.join('output_videos', 'review_transcript.txt')
    summary_path = os.path.join('output_videos', 'review_summary.txt')
    
    with open(full_transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript if transcript else "Không có lời thoại")
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(f"✅ Đã lưu bản tóm tắt review phim vào: {summary_path}")
    print("\n--- Nội dung tóm tắt ---")
    print(summary[:500] + "..." if len(summary) > 500 else summary)
    
    # Tạo voice-over cho bản tóm tắt nếu cần
    if summary.strip():  # Chỉ tạo voice-over nếu có nội dung tóm tắt
        voice_over_path = os.path.join('output_videos', 'review_voiceover.mp3')
        generate_voice_over(summary, voice_over_path)
    else:
        voice_over_path = None
    
    # Xóa file âm thanh tạm
    if os.path.exists(raw_audio_path):
        os.remove(raw_audio_path)
    if os.path.exists(filtered_audio_path):
        os.remove(filtered_audio_path)
    
    # Tạo video highlight các cảnh cao trào nếu có
    if climax_segments:
        try:
            highlight_path = os.path.join('output_videos', 'review_highlights.mp4')
            
            # Sử dụng CUDA nếu có thể
            if use_cuda and torch.cuda.is_available():
                from action_detection import fast_extract_segments_cuda
                if fast_extract_segments_cuda(input_path, highlight_path, climax_segments):
                    print(f"✅ Đã tạo video highlight các cảnh cao trào với CUDA: {highlight_path}")
            else:
                from action_detection import extract_segments_to_video
                if extract_segments_to_video(input_path, highlight_path, climax_segments, buffer_time=1.0):
                    print(f"✅ Đã tạo video highlight các cảnh cao trào: {highlight_path}")
        except Exception as e:
            print(f"⚠️ Không thể tạo video highlight: {e}")
    
    print("✅ Hoàn thành chế độ Review Phim!")
    return summary, voice_over_path

def process_short_mode(input_path, output_dir, model, device):
    """Chế độ Cắt Video Short - Phát hiện và cắt các cảnh cao trào"""
    print("\n=== CHẾ ĐỘ CẮT VIDEO SHORT ===")
    
    # Đường dẫn file output với tên phân biệt 
    summary_video_path = os.path.join(output_dir, "summary_output.mp4")
    climax_video_path = os.path.join(output_dir, "climax_output.mp4")
    combined_video_path = os.path.join(output_dir, "short.mp4")  # Output cuối cùng
    
    # Phát hiện các cảnh cao trào hoặc có người xuất hiện
    print("🔍 Đang phân tích video để tìm cảnh phù hợp cho video short...")
    
    # Tùy chọn phương pháp phát hiện
    print("\nChọn phương pháp phát hiện cảnh:")
    print("1. Phát hiện người xuất hiện")
    print("2. Phát hiện cảnh cao trào (chuyển động mạnh, thay đổi cảnh)")
    print("3. Kết hợp cả hai phương pháp")
    
    while True:
        try:
            detection_method = int(input("Lựa chọn của bạn (1-3): "))
            if 1 <= detection_method <= 3:
                break
            else:
                print("Vui lòng nhập số từ 1-3!")
        except ValueError:
            print("Vui lòng nhập số!")
            detection_method = 3  # Mặc định kết hợp
            break
    
    # Xác định thời lượng mong muốn cho video short
    print("\nNhập thời lượng tối đa mong muốn cho video short (giây):")
    try:
        max_duration = int(input("Thời lượng (thường là 15-60 giây): "))
    except ValueError:
        max_duration = 30
        print(f"Giá trị không hợp lệ, sử dụng giá trị mặc định: {max_duration}s")
    
    # Buffer time giữa các cảnh
    buffer_time = 1.0
    
    # Thực hiện phát hiện theo phương pháp đã chọn
    if detection_method == 1:
        # Phát hiện người xuất hiện
        success = extract_important_clips(input_path, summary_video_path, model, device, buffer_time, sample_rate=1)
        if success:
            import shutil
            shutil.copy(summary_video_path, combined_video_path)
            print(f"✅ Đã tạo video short dựa trên phát hiện người xuất hiện: {combined_video_path}")
        return success, combined_video_path
        
    elif detection_method == 2:
        # Phát hiện cảnh cao trào
        success = extract_climax_scenes(input_path, climax_video_path, buffer_time)
        if success:
            import shutil
            shutil.copy(climax_video_path, combined_video_path)
            print(f"✅ Đã tạo video short dựa trên phát hiện cảnh cao trào: {combined_video_path}")
        return success, combined_video_path
        
    else:
        # Kết hợp cả hai phương pháp
        print("Đang thực hiện phương pháp 1: Phát hiện người xuất hiện...")
        people_detected = extract_important_clips(input_path, summary_video_path, model, device, buffer_time, sample_rate=1)
        
        print("Đang thực hiện phương pháp 2: Phát hiện cảnh cao trào...")
        from action_detection import extract_climax_scenes_improved
        climax_detected = extract_climax_scenes_improved(input_path, climax_video_path, model, device, buffer_time=buffer_time, threshold_multiplier=1.0)
        
        # Xác định video nào sẽ được sử dụng làm output cuối cùng
        final_success = False
        
        # Kiểm tra và chọn video tốt nhất
        if people_detected and os.path.exists(summary_video_path) and os.stat(summary_video_path).st_size > 0:
            from moviepy.editor import VideoFileClip
            clip1 = VideoFileClip(summary_video_path)
            duration1 = clip1.duration
            clip1.close()
            
            if climax_detected and os.path.exists(climax_video_path) and os.stat(climax_video_path).st_size > 0:
                clip2 = VideoFileClip(climax_video_path)
                duration2 = clip2.duration
                clip2.close()
                
                # Nếu cả hai đều thành công, chọn video dài hơn hoặc ghép
                if abs(duration1 - duration2) < 10:
                    print(f"Đã tạo cả hai loại video có độ dài tương đương.")
                    
                    # Lấy video phát hiện người vì thường đa dạng hơn
                    import shutil
                    shutil.copy(summary_video_path, combined_video_path)
                    print(f"✅ Đã chọn video phát hiện người: {combined_video_path}")
                    final_success = True
                    
                elif duration1 > duration2 and duration1 <= max_duration:
                    import shutil
                    shutil.copy(summary_video_path, combined_video_path)
                    print(f"✅ Đã chọn video dài hơn (phát hiện người): {combined_video_path}")
                    final_success = True
                    
                elif duration2 <= max_duration:
                    import shutil
                    shutil.copy(climax_video_path, combined_video_path)
                    print(f"✅ Đã chọn video dài hơn (cảnh cao trào): {combined_video_path}")
                    final_success = True
                    
                else:
                    # Nếu cả hai đều dài hơn max_duration, lấy cái ngắn hơn
                    if duration1 <= duration2:
                        import shutil
                        shutil.copy(summary_video_path, combined_video_path)
                        print(f"✅ Đã chọn video ngắn hơn: {combined_video_path}")
                    else:
                        import shutil
                        shutil.copy(climax_video_path, combined_video_path)
                        print(f"✅ Đã chọn video ngắn hơn: {combined_video_path}")
                    final_success = True
                    
            else:
                # Chỉ phát hiện người thành công
                import shutil
                shutil.copy(summary_video_path, combined_video_path)
                print(f"✅ Chỉ phát hiện người thành công, đã chọn: {combined_video_path}")
                final_success = True
                
        elif climax_detected and os.path.exists(climax_video_path) and os.stat(climax_video_path).st_size > 0:
            # Chỉ phát hiện cảnh cao trào thành công
            import shutil
            shutil.copy(climax_video_path, combined_video_path)
            print(f"✅ Chỉ phát hiện cảnh cao trào thành công, đã chọn: {combined_video_path}")
            final_success = True
        
        else:
            print("❌ Không thể tạo video short với cả hai phương pháp.")
            final_success = False
        
        # Giữ lại các file trung gian nếu cần debug
        # Nếu không cần debug, hãy mở comment các dòng xóa file
        try:
            if final_success:  # Chỉ xóa nếu đã tạo thành công file output cuối cùng
                if os.path.exists(summary_video_path):
                    print(f"⚠️ Kiểm tra file trước khi xóa: {summary_video_path}")
                    if "summary" in summary_video_path:
                        print(f"❌ Không xóa {summary_video_path}, vì đó là video tóm tắt!")
                    else:
                        os.remove(summary_video_path)
                        print(f"✅ Đã xóa {summary_video_path} vì không cần thiết.")
                if os.path.exists(climax_video_path):
                    print(f"⚠️ Kiểm tra file trước khi xóa: {climax_video_path}")
                    if "climax" in climax_video_path:
                        print(f"❌ Không xóa {climax_video_path}, vì đó là video cao trào!")
                    else:
                        os.remove(climax_video_path)
                        print(f"✅ Đã xóa {climax_video_path} vì không cần thiết.")
        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý file tạm: {e}")
    
    print("✅ Hoàn thành chế độ Cắt Video Short!")
    return final_success, combined_video_path

def process_review_mode_fast(input_path, output_path, device, ratio=0.3):
    """Phiên bản tối ưu tốc độ của chế độ Review Phim"""
    print("\n=== CHẾ ĐỘ REVIEW PHIM (TỐC ĐỘ CAO) ===")
    
    # Khởi tạo model Whisper
    whisper_model = initialize_whisper_model(device)
    
    # Trích xuất và xử lý âm thanh - song song với phân tích video
    raw_audio_path = os.path.join('output_videos', 'temp_audio_raw.wav')
    filtered_audio_path = os.path.join('output_videos', 'temp_audio.wav')
    extract_audio(input_path, raw_audio_path)
    
    # Khởi động phát hiện cao trào trong một thread riêng
    climax_segments = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Sử dụng phiên bản nhanh của phát hiện cao trào
        from action_detection import detect_climax_scenes_fast
        future = executor.submit(detect_climax_scenes_fast, input_path, None, device)
    
    # Tiếp tục xử lý âm thanh trong khi phát hiện cao trào chạy
    # Kiểm tra nhanh xem video có lời thoại không
    has_speech = has_dialogue(raw_audio_path)
    
    if has_speech:
        filter_audio(raw_audio_path, filtered_audio_path)
        
        # Nhận dạng giọng nói
        print("🔍 Phân tích nội dung lời thoại...")
        transcript = transcribe_audio_optimized(filtered_audio_path, whisper_model)
        transcript = clean_transcript(transcript)
    else:
        print("⚠️ Không phát hiện được lời thoại rõ ràng")
        transcript = ""
    
    # Lấy kết quả từ thread phát hiện cao trào (nếu đã hoàn thành)
    try:
        # Chờ kết quả với timeout để tránh chờ quá lâu
        climax_segments = future.result(timeout=60)
    except Exception as e:
        print(f"⚠️ Không thể hoàn thành phát hiện cao trào: {e}")
    
    # Tạo bản tóm tắt dựa trên lời thoại và/hoặc cao trào
    if transcript and len(transcript.split()) >= 5:
        # Lọc và tóm tắt nội dung lời thoại
        filtered_transcript = filter_irrelevant_content(transcript)
        summary = enhanced_summarize_transcript(filtered_transcript, ratio)
        
        # Thêm thông tin cao trào
        if climax_segments:
            summary += "\n\n## Các cảnh cao trào:\n"
            for i, (start, end) in enumerate(climax_segments):
                summary += f"- Cảnh cao trào {i+1}: {start:.1f}s - {end:.1f}s\n"
    else:
        if climax_segments:
            summary = "## Tóm tắt dựa trên phân tích hình ảnh:\n\n"
            summary += "Video có các cảnh cao trào sau:\n"
            for i, (start, end) in enumerate(climax_segments):
                summary += f"- Cảnh cao trào {i+1}: {start:.1f}s - {end:.1f}s\n"
        else:
            summary = "Không thể tạo tóm tắt do không có lời thoại và không phát hiện được cảnh cao trào."
    
    # Lưu bản tóm tắt
    summary_path = os.path.join('output_videos', 'review_summary.txt')
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    
    # Clean up
    if os.path.exists(raw_audio_path):
        os.remove(raw_audio_path)
    if os.path.exists(filtered_audio_path):
        os.remove(filtered_audio_path)
    
    print("✅ Hoàn thành chế độ Review Phim (tốc độ cao)!")
    return summary

def debug_info():
    """Hiển thị thông tin debug cho người dùng"""
    print("\n===== THÔNG TIN HỆ THỐNG =====")
    
    # Kiểm tra CPU
    import multiprocessing
    print(f"CPU: {multiprocessing.cpu_count()} cores")
    
    # Kiểm tra GPU
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"   - RAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        else:
            print("GPU: Not available")
    except Exception as e:
        print(f"GPU check error: {e}")
    
    # Kiểm tra RAM
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"RAM: {vm.total / 1e9:.1f} GB total, {vm.available / 1e9:.1f} GB available")
    except Exception as e:
        print(f"RAM check error: {e}")
    
    # Kiểm tra FFmpeg
    try:
        import subprocess
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            version_line = result.stdout.decode().split("\n")[0]
            print(f"FFmpeg: {version_line}")
        else:
            print("FFmpeg: Not found")
    except:
        print("FFmpeg: Not found or error")
        
    # Kiểm tra thư viện
    try:
        import importlib
        libraries = ["cv2", "whisper", "faster_whisper", "moviepy", "torch", "numpy"]
        for lib in libraries:
            try:
                module = importlib.import_module(lib)
                version = getattr(module, "__version__", "unknown")
                print(f"{lib}: {version}")
            except ImportError:
                print(f"{lib}: Not installed")
    except:
        pass
    
    print("=============================\n")

def estimate_processing_time(video_path, mode):
    """Ước tính thời gian xử lý dựa trên kích thước và thời lượng video"""
    try:
        # Lấy kích thước file
        file_size_gb = os.path.getsize(video_path) / 1e9
        
        # Lấy độ dài video
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_minutes = (total_frames / fps) / 60
        cap.release()
        
        # Ước tính thời gian xử lý
        if mode == "review":
            # Review mode: mostly audio processing
            estimated_minutes = duration_minutes * 0.5  # Khoảng 1/2 thời lượng video
        elif mode == "short":
            # Short mode: heavy video processing
            estimated_minutes = duration_minutes * 0.3  # Khoảng 1/3 thời lượng video
        else:
            # Both modes
            estimated_minutes = duration_minutes * 0.6  # Khoảng 3/5 thời lượng video
            
        # Điều chỉnh theo kích thước file và RAM
        if file_size_gb > 1:
            estimated_minutes *= (1 + file_size_gb * 0.1)  # Thêm 10% cho mỗi GB
            
        # Round up
        estimated_minutes = int(estimated_minutes) + 1
        
        print(f"ℹ️ Thông tin video: {duration_minutes:.1f} phút, {file_size_gb:.2f} GB")
        print(f"⏱️ Ước tính thời gian xử lý: khoảng {estimated_minutes} phút")
        
        return estimated_minutes
    except Exception as e:
        print(f"Không thể ước tính thời gian xử lý: {e}")
        return None

def process_review_mode_with_timeout(input_path, output_path, device, model=None, ratio=0.3, timeout=300,
                                   use_faster_whisper=True, use_cuda=True):
    """Phiên bản process_review_mode với timeout để tránh xử lý quá lâu"""
    import threading
    import time
    
    # Import các hàm cần thiết từ các module khác
    try:
        from action_detection import fast_motion_detection, fast_motion_detection_with_decord
        from action_detection import fast_extract_segments, fast_extract_segments_cuda
    except ImportError:
        print("⚠️ Không thể import các hàm từ action_detection")
    
    try:
        from summary_generator import filter_irrelevant_content, enhanced_summarize_transcript
    except ImportError:
        print("⚠️ Không thể import các hàm từ summary_generator")
        # Phương pháp dự phòng
        def filter_irrelevant_content(text): return text
        def enhanced_summarize_transcript(text, ratio): return text[:int(len(text)*ratio)]
    
    # Biến dùng để lưu kết quả và kiểm tra timeout
    result = {"completed": False, "summary": "", "voice_over_path": None}
    timeout_occurred = False
    
    # Hàm thực thi chính - chạy trong thread riêng
    def process_with_timeout():
        try:
            # Phần 1: Xử lý audio (cần thiết)
            raw_audio_path = os.path.join('output_videos', 'temp_audio_raw.wav')
            filtered_audio_path = os.path.join('output_videos', 'temp_audio.wav')
            extract_audio(input_path, raw_audio_path)
            filter_audio(raw_audio_path, filtered_audio_path)
            
            # Sử dụng phương pháp song song để chuyển đổi âm thanh nhanh hơn
            print("🚀 Sử dụng phương pháp chuyển đổi âm thanh nhanh...")
            transcript = ""
            
            if use_faster_whisper:
                # Dùng phương pháp xử lý với Faster-Whisper
                try:
                    from audio_processing import transcribe_audio_faster
                    whisper_result = transcribe_audio_faster(filtered_audio_path)
                    if whisper_result and whisper_result["text"]:
                        transcript = whisper_result["text"]
                        from audio_processing import clean_transcript
                        transcript = clean_transcript(transcript)
                        print("✅ Chuyển đổi âm thanh thành công với Faster-Whisper")
                    else:
                        raise Exception("Faster-Whisper không trả về kết quả")
                except Exception as e:
                    print(f"⚠️ Lỗi khi dùng Faster-Whisper: {e}")
                    use_faster_whisper = False
            
            if not transcript and not use_faster_whisper:
                # Thử với phương pháp song song
                try:
                    from audio_processing import transcribe_audio_parallel
                    whisper_result = transcribe_audio_parallel(
                        filtered_audio_path, 
                        chunk_length_sec=15,
                        max_workers=4
                    )
                    if whisper_result and whisper_result["text"]:
                        transcript = whisper_result["text"]
                        from audio_processing import clean_transcript
                        transcript = clean_transcript(transcript)
                        print("✅ Chuyển đổi âm thanh thành công với phương pháp song song")
                except Exception as e:
                    print(f"⚠️ Lỗi khi dùng phương pháp song song: {e}")
                    
                    # Khởi tạo model Whisper và fallback
                    print("Chuyển sang whisper thông thường...")
                    whisper_model = initialize_whisper_model(device)
                    transcript = transcribe_audio_optimized(filtered_audio_path, whisper_model)
                    from audio_processing import clean_transcript
                    transcript = clean_transcript(transcript)
            
            # Phần 2: Tóm tắt văn bản và phân tích video
            if transcript and len(transcript.split()) >= 10:
                filtered_transcript = filter_irrelevant_content(transcript)
                summary = enhanced_summarize_transcript(filtered_transcript, ratio)
            else:
                summary = "Không phát hiện lời thoại có nghĩa trong video."
            
            # Phần 3: Phân tích video (thử nếu còn thời gian)
            try:
                segments = []
                # Sử dụng phiên bản nhanh để phát hiện chuyển động
                if use_cuda and 'fast_motion_detection_with_decord' in globals():
                    print("🚀 Sử dụng GPU để phát hiện chuyển động...")
                    segments = fast_motion_detection_with_decord(input_path)
                elif 'fast_motion_detection' in globals():
                    segments = fast_motion_detection(input_path)
                else:
                    print("⚠️ Không tìm thấy hàm phát hiện chuyển động")
                
                # Thêm thông tin về các đoạn chuyển động vào summary
                if segments:
                    summary += "\n\n## Các đoạn có chuyển động mạnh:\n"
                    for i, (start, end) in enumerate(segments):
                        summary += f"- Đoạn {i+1}: {start:.1f}s - {end:.1f}s (thời lượng: {end-start:.1f}s)\n"
                        
                # Tạo video highlight nếu có segments
                if segments and model is not None:
                    highlight_path = os.path.join('output_videos', 'review_highlights.mp4')
                    extracted = False
                    
                    if use_cuda and 'fast_extract_segments_cuda' in globals():
                        extracted = fast_extract_segments_cuda(input_path, highlight_path, segments)
                    elif 'fast_extract_segments' in globals():
                        extracted = fast_extract_segments(input_path, highlight_path, segments)
                        
                    if extracted:
                        print(f"✅ Đã tạo video highlight: {highlight_path}")
                    else:
                        print("⚠️ Không thể tạo video highlight")
                        
            except Exception as e:
                print(f"⚠️ Phân tích video gặp lỗi: {e}")
            
            # Lưu bản tóm tắt
            summary_path = os.path.join('output_videos', 'review_summary.txt')
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            
            # Tạo voice-over
            voice_over_path = os.path.join('output_videos', 'review_voiceover.mp3')
            generate_voice_over(summary, voice_over_path)
            
            # Dọn dẹp
            if os.path.exists(raw_audio_path):
                os.remove(raw_audio_path)
            if os.path.exists(filtered_audio_path):
                os.remove(filtered_audio_path)
            
            # Lưu kết quả vào biến toàn cục để truy cập từ thread chính
            result["completed"] = True
            result["summary"] = summary
            result["voice_over_path"] = voice_over_path
            
        except Exception as e:
            print(f"❌ Lỗi trong quá trình xử lý: {e}")
            summary = f"Đã xảy ra lỗi trong quá trình xử lý: {e}"
            summary_path = os.path.join('output_videos', 'review_summary.txt')
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            
            result["summary"] = summary
    
    # Tạo và chạy thread xử lý
    processing_thread = threading.Thread(target=process_with_timeout)
    processing_thread.daemon = True  # Cho phép thoát ứng dụng nếu thread còn chạy
    processing_thread.start()
    
    # Chờ thread trong khoảng thời gian timeout
    start_time = time.time()
    while processing_thread.is_alive():
        processing_thread.join(timeout=0.5)  # Kiểm tra mỗi 0.5 giây
        if time.time() - start_time > timeout:
            timeout_occurred = True
            print(f"⚠️ Quá thời gian xử lý {timeout} giây, tiến trình sẽ tiếp tục nhưng kết quả sẽ được trả về ngay")
            break
    
    # Xử lý timeout
    if timeout_occurred:
        print("⚠️ Quá thời gian xử lý, trả về kết quả đã có")
        # Trả về kết quả tạm thời nếu có
        summary_path = os.path.join('output_videos', 'review_summary.txt')
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = f.read()
        else:
            summary = "Xử lý timeout, không thể tạo tóm tắt đầy đủ."
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
        
        voice_over_path = os.path.join('output_videos', 'review_voiceover.mp3')
        if not os.path.exists(voice_over_path):
            generate_voice_over(summary, voice_over_path)
            
        return summary, voice_over_path
    
    # Trả về kết quả từ thread xử lý
    return result["summary"], result["voice_over_path"]

def setup_yolo8_model(device, task="detect"):
    """Setup YOLOv8 model for detection or pose estimation"""
    try:
        from action_detection import initialize_yolo8_model
        model_type = "yolov8s.pt" if task == "detect" else "yolov8s-pose.pt"
        model = initialize_yolo8_model(device, model_type, task)
        return model
    except Exception as e:
        print(f"⚠️ Failed to set up YOLOv8 model: {e}")
        return None

def process_short_mode_yolo8(input_path, output_dir, device="cuda", mode=1, 
                           target_duration=30.0, buffer_time=1.0):
    """Enhanced short video clipping with YOLOv8"""
    
    print(f"\n=== ENHANCED SHORT VIDEO MODE (YOLOv8) ===")
    
    # Kiểm tra rõ ràng device
    if device == "cuda" and torch.cuda.is_available():
        print(f"🚀 Sử dụng GPU cho xử lý video: {torch.cuda.get_device_name(0)}")
        # Đảm bảo các biến môi trường được đặt
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['FORCE_CUDA'] = '1'
    else:
        print("⚠️ CUDA không khả dụng hoặc không được yêu cầu, sử dụng CPU")
    
    # Set up output paths
    if mode == 1:
        output_path = os.path.join(output_dir, "single_action_short.mp4")
        print("📊 Mode: Single most intense action moment")
    else:
        output_path = os.path.join(output_dir, "multi_action_short.mp4")
        print(f"📊 Mode: Multiple action scenes (target: {target_duration:.1f}s)")
    
    # Initialize YOLOv8
    yolo_model = setup_yolo8_model(device)
    
    # Code tiếp theo không thay đổi...

    if yolo_model is None:
        print("⚠️ Failed to initialize YOLOv8, using fallback method")
        from action_detection import detect_climax_scenes_fast, fast_extract_segments
        
        # Use fallback method
        segments = detect_climax_scenes_fast(input_path, None, device)
        if segments:
            success = fast_extract_segments(input_path, output_path, segments, use_cuda=(device=="cuda"))
            return success, output_path
        else:
            print("❌ No action scenes detected")
            return False, None
    
    # Process based on mode
    if mode == 1:
        # Single most intense moment
        from action_detection import create_single_action_short
        success = create_single_action_short(
            input_path, output_path, yolo_model, device, 
            buffer_time=buffer_time, 
            min_duration=5.0,
            max_duration=15.0
        )
    else:
        # Multiple action moments
        from action_detection import create_multi_action_short
        success = create_multi_action_short(
            input_path, output_path, yolo_model, device,
            target_duration=target_duration,
            max_scenes=5,
            buffer_time=buffer_time
        )
    
    # Clean up GPU memory
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if success:
        print(f"✅ Short video created successfully: {output_path}")
        return True, output_path
    else:
        print("❌ Failed to create short video")
        return False, None

# Update the main function to use the new YOLOv8 capabilities
def debug_gpu():
    """Kiểm tra chi tiết GPU và tạo báo cáo"""
    print("\n===== THÔNG TIN GPU CHI TIẾT =====")
    
    # Kiểm tra xem PyTorch có thể tìm thấy CUDA không
    has_cuda_support = torch.cuda.is_available()
    print(f"PyTorch nhận diện CUDA: {has_cuda_support}")
    
    # Kiểm tra thông tin GPU qua các phương pháp khác nhau
    try:
        import subprocess
        # Trên Windows, dùng nvidia-smi
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("\nThông tin từ nvidia-smi:")
            print(result.stdout.split('\n')[0])
            print(result.stdout.split('\n')[1])
            
            # CUDA driver version
            import re
            match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
            if match:
                cuda_version = match.group(1)
                print(f"CUDA Driver Version: {cuda_version}")
                
                # So sánh với torch.version.cuda
                torch_cuda_version = torch.version.cuda
                print(f"PyTorch CUDA Version: {torch_cuda_version}")
                
                if torch_cuda_version != cuda_version:
                    print(f"⚠️ Phiên bản CUDA không khớp! Driver: {cuda_version}, PyTorch: {torch_cuda_version}")
                    print("Điều này có thể khiến PyTorch không nhận diện GPU")
            else:
                print("⚠️ Không tìm thấy phiên bản CUDA trong nvidia-smi")
        else:
            print("⚠️ Không thể chạy nvidia-smi. GPU có thể không được cài đặt đúng cách")
            
    except Exception as e:
        print(f"⚠️ Lỗi khi kiểm tra GPU: {e}")
    
    print("================================\n")

import os
import sys
import torch
from utils.gpu_utils import force_gpu_usage, get_gpu_info, debug_gpu_detection

def main():
    print("===== TOOLFILM AI =====")
    
    # THÊM: Buộc nhận diện GPU ở đây, đầu tiên
    force_gpu_usage()
    
    # Debug GPU/CPU
    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        print(f"🚀 Đã phát hiện GPU: {gpu_info['devices'][0]['name']}")
        print(f"   Bộ nhớ: {gpu_info['devices'][0]['total_memory_gb']:.1f} GB")
    else:
        print("⚠️ Không phát hiện GPU, sẽ sử dụng CPU")
        print("   Kiểm tra cài đặt CUDA và trình điều khiển NVIDIA")
        
    # Kiểm tra xem FORCE_CUDA có được cài đặt không
    force_cuda = os.environ.get('FORCE_CUDA', '0')
    if force_cuda == '1':
        print("✅ Chế độ FORCE_CUDA được kích hoạt")
    
    # Kiểm tra ULTRALYTICS_DEVICE đã được cài đặt chưa
    ultralytics_device = os.environ.get('ULTRALYTICS_DEVICE', 'None')
    print(f"✓ ULTRALYTICS_DEVICE = {ultralytics_device}")
    
    # Sử dụng module yolo_utils thay vì torch.hub trực tiếp
    try:
        from utils.yolo_utils import initialize_yolo5_model
        
        # Chuyển đến device phù hợp
        device = "cuda" if gpu_info["available"] else "cpu"
        yolo_model = initialize_yolo5_model(weights="models/yolov5s.pt", device=device)
        
        if yolo_model is not None:
            print("✅ Khởi tạo model YOLOv5 thành công")
        else:
            print("❌ Không thể khởi tạo model YOLOv5")
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo model: {e}")
    
    # Thử khởi tạo YOLOv8 để kiểm tra
    try:
        from yolo8_detection import initialize_yolo8_model
        yolo8_model = initialize_yolo8_model(device="cuda")
        if yolo8_model is not None:
            print("✅ Khởi tạo model YOLOv8 thành công")
            print(f"   Device: {next(yolo8_model.parameters()).device}")
        else:
            print("❌ Không thể khởi tạo model YOLOv8")
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo YOLOv8: {e}")
    
    # Code tiếp theo...
    # ...existing code...

if __name__ == "__main__":
    # Check if the YOLOv8 GPU fix is being applied
    if os.environ.get('ULTRALYTICS_DEVICE') == 'cuda:0':
        print("✅ YOLOv8 GPU fix is active")
    else:
        print("⚠️ YOLOv8 GPU fix is not active. Consider running with run_yolo8_gpu_fix.bat instead")
    
    main()