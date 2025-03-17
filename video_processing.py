import cv2  # Import này thiếu trong file
import torch
import os
import numpy as np
import subprocess
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm
import multiprocessing
from functools import partial

try:
    import decord
    from decord import VideoReader, cpu, gpu
    DECORD_AVAILABLE = True
    
    # Buộc sử dụng GPU nếu có
    if torch.cuda.is_available():
        try:
            decord.bridge.set_bridge('torch')
            print("✅ Decord sẽ sử dụng GPU với PyTorch bridge")
            # Khởi tạo GPU context sớm để kiểm tra
            test_ctx = decord.gpu(0)
            print("✅ Decord GPU context khởi tạo thành công")
        except Exception as e:
            print(f"⚠️ Không thể khởi tạo Decord GPU: {e}")
            decord.bridge.set_bridge('native')
            print("⚠️ Fallback to native bridge")
    else:
        decord.bridge.set_bridge('native')
        print("✅ Decord sẽ sử dụng CPU với native bridge")
except ImportError:
    DECORD_AVAILABLE = False
    print("⚠️ Decord library not found. Using slower OpenCV for frame extraction.")

def detect_objects(model, frame, frame_idx, device, use_opencv_cuda=False):
    """Nhận diện vật thể trong frame."""
    # Chuyển đổi màu và resize
    if isinstance(frame, np.ndarray):
        if use_opencv_cuda:
            # Sử dụng OpenCV CUDA
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gpu_frame_resized = cv2.cuda.resize(gpu_frame, (640, 640))
            frame_resized = gpu_frame_resized.download()
        else:
            frame_resized = cv2.resize(frame, (640, 640))
    else:
        frame = np.array(frame)
        if use_opencv_cuda:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gpu_frame_resized = cv2.cuda.resize(gpu_frame, (640, 640))
            frame_resized = gpu_frame_resized.download()
        else:
            frame_resized = cv2.resize(frame, (640, 640))
    
    # Thực hiện phát hiện với batch processing để tối ưu hóa GPU model
    results = model(frame_resized)
    detections = results.xyxy[0].cpu().numpy()  # Luôn chuyển về CPU để xử lý numpy
    
    for detection in detections:
        # Chỉ lấy người (class 0 trong COCO dataset)
        if int(detection[5]) == 0 and float(detection[4]) > 0.4:  # Thêm ngưỡng tin cậy
            return (True, frame_idx, float(detection[4]))  # Trả về tuple với điểm tin cậy
    return (False, frame_idx, 0.0)

def detect_objects_optimized(model, frames, device, batch_size=8):
    """Process multiple frames at once for efficient object detection"""
    results = []
    
    # Process frames in batches to make better use of the GPU
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        
        # Convert frames to format expected by YOLO
        processed_batch = []
        for frame in batch:
            # Resize to expected input size
            resized = cv2.resize(frame, (640, 640))
            processed_batch.append(resized)
        
        # Run detection on batch
        batch_results = model(processed_batch)
        
        # Extract detection results
        for j, detections in enumerate(batch_results.xyxy):
            frame_idx = i + j
            has_person = False
            confidence = 0.0
            
            for det in detections:
                # Check if detection is a person (class 0 in COCO)
                if int(det[5]) == 0 and float(det[4]) > 0.4:
                    has_person = True
                    confidence = max(confidence, float(det[4]))
            
            results.append((has_person, frame_idx, confidence))
    
    return results

def detect_motion(prev_frame, current_frame, threshold=2.0):
    """Phát hiện chuyển động giữa hai frame liên tiếp"""
    if prev_frame is None or current_frame is None:
        return False, 0
        
    try:
        # Chuyển đổi sang grayscale
        prev_gray = cv2.cvtColor(np.array(prev_frame).astype('uint8'), cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(np.array(current_frame).astype('uint8'), cv2.COLOR_RGB2GRAY)
        
        # Tính toán optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Tính magnitude và angle
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_magnitude = np.mean(mag)
        
        # Trả về True nếu chuyển động đủ lớn
        return motion_magnitude > threshold, motion_magnitude
    except Exception as e:
        print(f"Lỗi khi phát hiện chuyển động: {e}")
        return False, 0

def process_frame(t, clip, batch_start, model, device, use_opencv_cuda=False):
    """Xử lý một frame trong video"""
    try:
        frame = clip.get_frame(t)
        result = detect_objects(model, frame, t, device, use_opencv_cuda)
        return result
    except Exception as e:
        print(f"Lỗi khi xử lý frame tại giây {t}: {e}")
        return (False, t, 0.0)

def extract_important_clips(input_path, output_path, model, device, buffer_time=3, sample_rate=1):
    """Cắt đoạn video có vật thể quan trọng và chuyển động."""
    # Check if we can use the optimized version with Decord
    if DECORD_AVAILABLE and device.type == 'cuda':
        print("🚀 Sử dụng Decord để trích xuất frame nhanh hơn")
        return extract_important_clips_optimized(input_path, output_path, model, device, buffer_time, sample_rate)
    
    clip = VideoFileClip(input_path)
    fps = clip.fps
    duration = clip.duration
    important_times = []
    
    print("Bắt đầu quét video để tìm khung hình có người và chuyển động...")
    total_seconds = int(duration)
    
    # Xử lý theo đoạn để giảm tải bộ nhớ GPU
    batch_size = min(300, total_seconds)  # Xử lý tối đa 5 phút mỗi lần
    
    # Xác định số lượng worker cho multiprocessing
    num_workers = min(os.cpu_count(), 4) if torch.cuda.is_available() else max(1, os.cpu_count() - 1)
    print(f"Sử dụng {num_workers} worker cho xử lý song song")
    
    # Xác định có sử dụng OpenCV CUDA không
    use_opencv_cuda = False
    if device.type == 'cuda':
        try:
            cv2.cuda.getCudaEnabledDeviceCount()
            use_opencv_cuda = True
            print("✅ Kích hoạt OpenCV CUDA để tăng tốc xử lý hình ảnh")
        except:
            print("⚠️ OpenCV CUDA không khả dụng")
    
    # Thông tin phát hiện người và chuyển động
    person_detections = []
    motion_scores = []
    previous_frame = None
    
    for batch_start in range(0, total_seconds, batch_size):
        batch_end = min(batch_start + batch_size, total_seconds)
        print(f"Xử lý đoạn: {batch_start}s - {batch_end}s")
        
        # Giải phóng bộ nhớ GPU nếu cần
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        # Chuẩn bị danh sách thời điểm cần xử lý
        times_to_process = list(range(batch_start, batch_end, sample_rate))
                
        # Sử dụng multiprocessing nếu không dùng GPU hoặc nếu dùng GPU với xử lý song song
        if device.type == 'cpu' or (device.type == 'cuda' and torch.cuda.device_count() > 1):
            # Xử lý song song với multiprocessing
            with multiprocessing.Pool(processes=num_workers) as pool:
                process_func = partial(process_frame, clip=clip, batch_start=batch_start, 
                                      model=model, device=device, use_opencv_cuda=False)
                results = list(tqdm(
                    pool.imap(process_func, times_to_process),
                    total=len(times_to_process),
                    desc="Xử lý video"
                ))
                
                # Thu thập kết quả
                for has_person, frame_idx, confidence in results:
                    # Lưu thông tin phát hiện người
                    person_detections.append((frame_idx, has_person, confidence))
        else:
            # Xử lý tuần tự khi dùng GPU
            for t in tqdm(times_to_process, desc="Xử lý video"):
                try:
                    frame = clip.get_frame(t)
                    
                    # Phát hiện người trong frame với GPU acceleration
                    has_person, _, confidence = detect_objects(model, frame, t, device, use_opencv_cuda)
                    person_detections.append((t, has_person, confidence))
                    
                    # Tính toán chuyển động giữa các frame liên tiếp
                    if previous_frame is not None:
                        has_motion, motion_magnitude = detect_motion(previous_frame, frame)
                        if has_motion:
                            motion_scores.append((t, motion_magnitude))
                    
                    previous_frame = frame
                    
                except Exception as e:
                    print(f"Lỗi khi xử lý frame tại giây {t}: {e}")
    
    # Phân tích kết quả phát hiện người
    important_segments = []
    current_segment = None
    
    # Ngưỡng phát hiện chuyển động
    motion_threshold = 2.0 if motion_scores else 0
    if motion_scores:
        # Tính ngưỡng thích ứng: 80% của trung vị
        motion_values = [score for _, score in motion_scores]
        motion_threshold = np.median(motion_values) * 0.8
        print(f"Ngưỡng chuyển động tự động: {motion_threshold:.2f}")
    
    # Sắp xếp các phát hiện theo thời gian
    person_detections.sort(key=lambda x: x[0])
    
    # Tạo các đoạn có người xuất hiện
    for t, has_person, confidence in person_detections:
        if has_person:
            # Kiểm tra mức độ chuyển động nếu có dữ liệu
            has_motion = True
            if motion_scores:
                # Tìm điểm chuyển động gần nhất với thời điểm t
                nearest_motion = min(motion_scores, key=lambda x: abs(x[0] - t))
                if nearest_motion[1] < motion_threshold:
                    has_motion = False
            
            # Nếu có người và có chuyển động đủ lớn
            if has_motion:
                if current_segment is None:
                    current_segment = [t, t]
                else:
                    # Mở rộng đoạn hiện tại
                    current_segment[1] = t
            
        elif current_segment is not None:
            # Kết thúc đoạn hiện tại
            if current_segment[1] - current_segment[0] >= 2:  # Đoạn phải dài ít nhất 2 giây
                important_segments.append(current_segment.copy())
            current_segment = None
    
    # Thêm đoạn cuối nếu còn
    if current_segment is not None and current_segment[1] - current_segment[0] >= 2:
        important_segments.append(current_segment)
    
    # Ghép các đoạn gần nhau thông minh hơn
    merged_segments = []
    max_merge_gap = 5  # Khoảng cách tối đa để ghép (giây)
    
    if important_segments:
        # Sắp xếp theo thời gian bắt đầu
        important_segments.sort(key=lambda x: x[0])
        
        current_merged = [max(0, important_segments[0][0] - buffer_time), 
                         min(duration, important_segments[0][1] + buffer_time)]
        
        for i in range(1, len(important_segments)):
            segment_start = max(0, important_segments[i][0] - buffer_time)
            segment_end = min(duration, important_segments[i][1] + buffer_time)
            
            # Nếu đoạn hiện tại gần với đoạn đã ghép
            if segment_start - current_merged[1] <= max_merge_gap:
                # Ghép đoạn nếu khoảng cách giữa chúng không quá lớn
                current_merged[1] = segment_end
            else:
                # Tính toán độ dài đoạn đã ghép
                merged_duration = current_merged[1] - current_merged[0]
                
                # Chỉ giữ lại đoạn đủ dài và có đủ chuyển động
                if merged_duration >= 3:  # Đoạn ghép phải dài ít nhất 3 giây
                    merged_segments.append(current_merged.copy())
                
                # Bắt đầu đoạn ghép mới
                current_merged = [segment_start, segment_end]
        
        # Xử lý đoạn cuối cùng
        final_duration = current_merged[1] - current_merged[0]
        if final_duration >= 3:
            merged_segments.append(current_merged)
    
    print(f"Tìm thấy {len(merged_segments)} đoạn quan trọng, đang xuất video...")
    
    # Thêm phân tích xem các đoạn có chiếm bao nhiêu % tổng thời gian
    if merged_segments:
        total_selected_duration = sum(end - start for start, end in merged_segments)
        percentage = (total_selected_duration / duration) * 100
        print(f"Tổng thời gian được chọn: {total_selected_duration:.1f}s ({percentage:.1f}% của video gốc)")
    
    # Xuất video
    if merged_segments:
        final_clips = []
        for i, (start, end) in enumerate(merged_segments):
            print(f"Đoạn {i+1}/{len(merged_segments)}: {start:.1f}s - {end:.1f}s (độ dài: {end-start:.1f}s)")
            final_clips.append(clip.subclip(start, end))
        
        print("Đang xuất file video...")
        # Dùng threads để tăng tốc xuất file
        final_video = concatenate_videoclips(final_clips)
        final_video.write_videofile(output_path, codec="libx264", fps=clip.fps, threads=min(4, os.cpu_count()))
        print(f"✅ Xuất video thành công! File: {output_path}")
        return True
    else:
        print("⛔ Không có cảnh quan trọng nào, không xuất video.")
        return False

def extract_important_clips_optimized(input_path, output_path, model, device, buffer_time=3, sample_rate=1):
    """Optimized version of extract_important_clips using Decord for frame extraction"""
    print("🚀 Using accelerated frame extraction for object detection")
    
    if DECORD_AVAILABLE:
        # Use GPU if available
        ctx = decord.gpu(0) if torch.cuda.is_available() else decord.cpu()
        try:
            vr = decord.VideoReader(input_path, ctx=ctx)
            
            fps = vr.get_avg_fps()
            frame_count = len(vr)
            duration = frame_count / fps
            
            # Sample frames at the specified rate (default: 1 frame per second)
            frame_indices = list(range(0, frame_count, int(fps * sample_rate)))
            
            print(f"Analyzing {len(frame_indices)} frames out of {frame_count}...")
            
            # Extract frames in batches
            batch_size = 16
            all_frames = []
            
            for i in tqdm(range(0, len(frame_indices), batch_size), desc="Extracting frames"):
                batch_indices = frame_indices[i:i+batch_size]
                # Get frames directly as torch tensors
                if torch.cuda.is_available():
                    batch_frames = vr.get_batch(batch_indices).cuda()
                else:
                    batch_frames = vr.get_batch(batch_indices)
                    
                # Convert from RGB to BGR for OpenCV compatibility
                for j in range(len(batch_frames)):
                    frame = batch_frames[j].cpu().numpy()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    all_frames.append(frame)
                
                # Free up memory
                del batch_frames
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Process frames for object detection
            detection_results = detect_objects_optimized(model, all_frames, device)
            
            # Process detection results
            important_indices = []
            for has_person, frame_idx, confidence in detection_results:
                if has_person:
                    important_indices.append(frame_idx)
            
            # Convert frame indices to timestamps
            important_times = [frame_indices[idx] / fps for idx in important_indices]
            
            # Group adjacent frames into segments
            segments = []
            if important_times:
                start_time = important_times[0]
                prev_time = important_times[0]
                
                for t in important_times[1:]:
                    # If gap is too big, start new segment
                    if t - prev_time > 2.0:  # 2 second gap
                        segments.append([start_time, prev_time])
                        start_time = t
                    prev_time = t
                
                # Add the last segment
                segments.append([start_time, prev_time])
            
            # Add buffer and merge close segments
            merged_segments = []
            if segments:
                for start, end in segments:
                    start_with_buffer = max(0, start - buffer_time)
                    end_with_buffer = min(duration, end + buffer_time)
                    
                    # Merge with previous segment if close enough
                    if merged_segments and start_with_buffer - merged_segments[-1][1] < 3.0:
                        merged_segments[-1][1] = end_with_buffer
                    else:
                        merged_segments.append([start_with_buffer, end_with_buffer])
            
            # Create clips and output video
            if merged_segments:
                clip = VideoFileClip(input_path)
                clips = []
                
                for i, (start, end) in enumerate(merged_segments):
                    if end - start >= 2.0:  # Only use segments of at least 2 seconds
                        print(f"Segment {i+1}: {start:.1f}s - {end:.1f}s")
                        clips.append(clip.subclip(start, end))
                
                if clips:
                    final_video = concatenate_videoclips(clips)
                    final_video.write_videofile(output_path, codec="libx264")
                    final_video.close()
                    clip.close()
                    
                    print(f"✅ Created short video with {len(clips)} important clips")
                    return True
                else:
                    clip.close()
                    print("No valid segments found")
                    return False
            else:
                print("No important segments detected")
                return False
                
        except Exception as e:
            print(f"Error using Decord: {e}")
            # Fall back to original implementation
            return extract_important_clips(input_path, output_path, model, device, buffer_time, sample_rate)
    
    # Fall back to original implementation
    return extract_important_clips(input_path, output_path, model, device, buffer_time, sample_rate)

def detect_action(model, input_path, output_path, action_classes=['person'], threshold=0.4):
    """Nhận diện các hành động cụ thể trong video"""
    print(f"Đang phân tích video để tìm hành động: {', '.join(action_classes)}...")
    
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Danh sách các segment có action
    action_segments = []
    current_segment = None
    
    # Các chỉ số lớp cần tìm
    class_indices = []
    for action in action_classes:
        if action.lower() in model.names:
            class_indices.append(list(model.names.values()).index(action.lower()))
    
    print(f"Quét {frame_count} frames...")
    
    # Phân tích từng frame
    for frame_idx in tqdm(range(0, frame_count, int(fps/2))):  # Lấy mẫu 2 frame/giây
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Nhận diện đối tượng trong frame
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()
        
        action_detected = False
        
        # Kiểm tra xem có đối tượng cần tìm không
        for detection in detections:
            if int(detection[5]) in class_indices and detection[4] > threshold:
                action_detected = True
                break
                
        # Xử lý segment
        time_sec = frame_idx / fps
        
        if action_detected:
            if current_segment is None:  # Bắt đầu segment mới
                current_segment = [time_sec, time_sec]
            else:  # Mở rộng segment hiện tại
                current_segment[1] = time_sec
        elif current_segment is not None:
            # Kết thúc segment hiện tại nếu không phát hiện action
            action_segments.append(current_segment)
            current_segment = None
    
    # Thêm segment cuối cùng nếu còn
    if current_segment is not None:
        action_segments.append(current_segment)
    
    cap.release()
    
    # Thêm buffer và merge các segment gần nhau
    buffer_time = 1.0  # Thêm 1 giây trước và sau mỗi segment
    merged_segments = []
    
    for segment in action_segments:
        start_time = max(0, segment[0] - buffer_time)
        end_time = segment[1] + buffer_time
        
        if not merged_segments or start_time > merged_segments[-1][1]:
            merged_segments.append([start_time, end_time])
        else:
            merged_segments[-1][1] = end_time
    
    # Xuất video các segment
    if merged_segments:
        clip = VideoFileClip(input_path)
        final_clips = []
        
        for i, (start, end) in enumerate(merged_segments):
            print(f"Hành động {i+1}: {start:.1f}s - {end:.1f}s")
            final_clips.append(clip.subclip(start, end))
        
        # Ghép và xuất video
        print(f"Tạo video với {len(final_clips)} đoạn hành động...")
        final_video = concatenate_videoclips(final_clips)
        final_video.write_videofile(output_path, codec="libx264")
        clip.close()
        
        return True, merged_segments
    else:
        print(f"Không tìm thấy hành động {action_classes} trong video.")
        return False, []

def create_highlight_compilation(input_path, output_path, segments, max_duration=60):
    """Tạo video tổng hợp các phần hay nhất với thời lượng tối đa"""
    if not segments:
        print("Không có đoạn video nào để tạo highlight")
        return False
        
    print(f"Tạo video highlight với thời lượng tối đa {max_duration}s...")
    
    # Sắp xếp các segment theo mức độ quan trọng (giả sử là độ dài)
    segments_with_scores = [(start, end, end-start) for start, end in segments]
    segments_with_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Chọn các segment quan trọng nhất mà vẫn đảm bảo thời lượng tối đa
    selected_segments = []
    current_duration = 0
    
    for start, end, duration in segments_with_scores:
        if current_duration + duration <= max_duration:
            selected_segments.append((start, end))
            current_duration += duration
        else:
            # Cắt segment cuối nếu vượt quá thời lượng
            remaining_time = max_duration - current_duration
            if remaining_time > 5:  # Chỉ thêm nếu còn ít nhất 5 giây
                selected_segments.append((start, start + remaining_time))
            break
    
    # Sắp xếp lại theo thứ tự thời gian
    selected_segments.sort()
    
    # Tạo video highlight
    clip = VideoFileClip(input_path)
    highlight_clips = [clip.subclip(start, end) for start, end in selected_segments]
    
    if highlight_clips:
        final_video = concatenate_videoclips(highlight_clips)
        final_video.write_videofile(output_path, codec="libx264")
        clip.close()
        print(f"✅ Đã tạo video highlight thành công: {output_path}")
        return True
    else:
        print("⚠️ Không thể tạo video highlight")
        return False

def initialize_faster_whisper():
    try:
        from faster_whisper import WhisperModel
        
        print("Khởi tạo Faster Whisper model...")
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            
            if gpu_memory >= 8.0:
                model_size = "medium"
            else:
                model_size = "small"
            
            try:
                # Dùng float32 thay vì float16 cho GTX 1060
                print(f"🔍 GPU có {gpu_memory:.1}GB RAM, sử dụng model {model_size} với float32")
                model = WhisperModel(model_size, device="cuda", compute_type="float32")
                return model
            except:
                print(f"🔍 Thử sử dụng int8 với {model_size}")
                model = WhisperModel(model_size, device="cuda", compute_type="int8")
                return model
        else:
            print("⚠️ Không phát hiện GPU, sử dụng CPU với model small")
            model = WhisperModel("small", device="cpu", compute_type="int8")
            return model
    except Exception as e:
        print(f"⚠️ Lỗi khi khởi tạo Faster Whisper: {e}")
        return None

def extract_frames_with_decord(video_path, sample_rate=1, device_id=0):
    """Extract frames from video using Decord library"""
    try:
        # Use CPU context if device_id is -1 or GPU not available
        if device_id < 0 or not torch.cuda.is_available():
            ctx = decord.cpu()
        else:
            ctx = decord.gpu(device_id)
            
        # Create video reader
        vr = decord.VideoReader(video_path, ctx=ctx)
        fps = vr.get_avg_fps()
        frame_count = len(vr)
        
        # Calculate which frames to extract
        frame_indices = list(range(0, frame_count, int(fps * sample_rate)))
        timestamps = [idx / fps for idx in frame_indices]
        
        # Extract frames
        frames = vr.get_batch(frame_indices).asnumpy()
        
        return frames, timestamps
    except Exception as e:
        print(f"Error using Decord: {e}")
        return None, None

def extract_frames_with_cpp(video_path, sample_rate=1.0, resize_width=0, resize_height=0):
    """Extract frames from video using C++ extension if available"""
    try:
        from utils.gpu_utils import try_load_cpp_extensions
        cpp_extensions = try_load_cpp_extensions()
        
        if "video_extract" in cpp_extensions:
            print("Using C++ video frame extraction")
            video_extract = cpp_extensions["video_extract"]
            
            # Get video info first
            try:
                video_info = video_extract.get_video_info(video_path)
                print(f"Video info: {video_info.width}x{video_info.height}, {video_info.fps} fps, {video_info.duration} seconds")
            except Exception as e:
                print(f"Error getting video info: {e}")
                return None, None, None
            
            # Extract frames using C++ module
            frames, timestamps, fps = video_extract.extract_frames_with_sample_rate(
                video_path, sample_rate, resize_width, resize_height)
            
            return frames, timestamps, fps
    except Exception as e:
        print(f"Error using C++ video extraction: {e}")
    
    # Fallback to existing methods
    if DECORD_AVAILABLE:
        print("Falling back to Decord for frame extraction")
        return extract_frames_with_decord(video_path, sample_rate)
    else:
        print("Falling back to OpenCV for frame extraction")
        return extract_frames_with_opencv(video_path, sample_rate)

def concatenate_videos_cpp(input_files, output_file, codec="libx264"):
    """Concatenate videos using direct FFmpeg C++ API if available"""
    try:
        from utils.gpu_utils import try_load_cpp_extensions
        cpp_extensions = try_load_cpp_extensions()
        
        if "video_concat" in cpp_extensions:
            print("Using C++ FFmpeg API for video concatenation")
            video_concat = cpp_extensions["video_concat"]
            
            # Concatenate videos using C++ module
            success = video_concat.concatenate_videos(input_files, output_file, codec)
            if success:
                print(f"✅ Successfully concatenated videos to: {output_file}")
                return True
            else:
                print("❌ Failed to concatenate videos with C++ extension")
    except Exception as e:
        print(f"Error using C++ video concatenation: {e}")
    
    # Fallback to existing FFmpeg subprocess method
    print("Falling back to FFmpeg subprocess for concatenation")
    return concatenate_videos_ffmpeg(input_files, output_file, codec)

def extract_frames_with_opencv(video_path, sample_rate=1):
    """Extract frames using OpenCV at specified sample rate"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices based on sample rate
    frame_skip = int(fps * sample_rate)
    frame_indices = list(range(0, total_frames, frame_skip))
    
    frames = []
    timestamps = []
    
    for frame_idx in tqdm(frame_indices, desc="Extracting frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            timestamps.append(frame_idx / fps)
    
    cap.release()
    return np.array(frames), timestamps, fps

def concatenate_videos_ffmpeg(input_files, output_file, codec="libx264"):
    """Concatenate videos using FFmpeg subprocess"""
    try:
        # Prepare file list for FFmpeg
        with open("file_list.txt", "w") as f:
            for file in input_files:
                f.write(f"file '{file}'\n")
        
        # Run FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", "file_list.txt",
            "-c:v", codec,
            "-c:a", "aac",
            output_file
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        os.remove("file_list.txt")
        return True
    except Exception as e:
        print(f"Error concatenating videos with FFmpeg: {e}")
        return False