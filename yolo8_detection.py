import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip, concatenate_videoclips
import matplotlib.pyplot as plt
import time
import subprocess
import shutil
from pathlib import Path

# Add Decord support for accelerated frame extraction
try:
    import decord
    from decord import VideoReader, cpu, gpu
    DECORD_AVAILABLE = True
    
    # Buộc sử dụng GPU nếu có
    force_cuda = os.environ.get('FORCE_CUDA', '0') == '1'
    if torch.cuda.is_available() or force_cuda:
        try:
            decord.bridge.set_bridge('torch')
            print("✅ Decord sẽ sử dụng GPU với PyTorch bridge")
            # Khởi tạo GPU context sớm để kiểm tra
            try:
                test_ctx = decord.gpu(0)
                print("✅ Decord GPU context khởi tạo thành công")
            except Exception as e:
                print(f"⚠️ Không thể khởi tạo Decord GPU context thông thường: {e}")
                print("   Thử phương pháp thay thế...")
                # Alternative method: Force by setting environment variable
                os.environ["DECORD_USE_CUDA"] = "1"
                try:
                    test_ctx = decord.gpu(0)
                    print("✅ Decord GPU context khởi tạo thành công qua biến môi trường")
                except Exception as e2:
                    print(f"⚠️ Vẫn không thể khởi tạo Decord GPU: {e2}")
                    decord.bridge.set_bridge('native')
                    print("⚠️ Fallback to native bridge")
        except Exception as e:
            print(f"⚠️ Không thể thiết lập PyTorch bridge: {e}")
            decord.bridge.set_bridge('native')
            print("⚠️ Fallback to native bridge")
    else:
        decord.bridge.set_bridge('native')
        print("✅ Decord sẽ sử dụng CPU với native bridge")
except ImportError:
    DECORD_AVAILABLE = False
    print("⚠️ Decord library not found. Using slower OpenCV for frame extraction.")

def initialize_yolo8_model(device="cuda", model_size="s", task="detect"):
    """Initialize a YOLOv8 model with specified size and task"""
    from ultralytics import YOLO
    
    # Kiểm tra rõ ràng xem CUDA có khả dụng không và thông báo
    cuda_available = torch.cuda.is_available()
    force_cuda = os.environ.get('FORCE_CUDA', '0') == '1'
    
    if not cuda_available:
        if force_cuda:
            print("🔄 FORCE_CUDA được bật, nhưng PyTorch không phát hiện GPU")
            print("⚠️ Nguyên nhân: PyTorch không được cài đặt với hỗ trợ CUDA")
            print("🔧 Giải pháp: Chạy 'run_cuda_fix.bat' để cài đặt lại PyTorch với CUDA")
            device = "cpu"  # Fallback to CPU since CUDA isn't available in PyTorch
        else:
            print("⚠️ CUDA không khả dụng, sử dụng CPU. Để buộc dùng GPU, hãy cài đặt FORCE_CUDA=1")
            device = "cpu"
    else:
        print(f"🚀 CUDA khả dụng, sử dụng GPU: {torch.cuda.get_device_name(0)}")
        # Đảm bảo cache GPU được giải phóng
        torch.cuda.empty_cache()
    
    # Xác định đường dẫn model
    if task == "detect":
        model_path = f"yolov8{model_size}.pt"
    elif task == "pose":
        model_path = f"yolov8{model_size}-pose.pt"
    else:
        raise ValueError(f"Không hỗ trợ loại task: {task}")
    
    # Kiểm tra thư mục models
    models_dir = os.path.join(os.getcwd(), "models")
    if os.path.exists(os.path.join(models_dir, model_path)):
        model_path = os.path.join(models_dir, model_path)
        print(f"Using model from models directory: {model_path}")
    
    print(f"🔍 Initializing YOLOv8 model for {task}...")
    
    try:
        # Tải model với cài đặt device rõ ràng
        model = YOLO(model_path)
        
        # Đặt thiết bị rõ ràng cho model - ĐÂY LÀ DÒNG QUAN TRỌNG
        model.to(device)
        
        # Print thông báo về thiết bị đang sử dụng
        try:
            model_device = next(model.parameters()).device
            print(f"✅ YOLOv8 đã được tải lên thiết bị: {model_device}")
        except Exception as e:
            print(f"⚠️ Không thể xác định thiết bị model: {e}")
        
        return model
    except Exception as e:
        print(f"❌ Lỗi khởi tạo YOLOv8: {e}")
        if "Torch not compiled with CUDA enabled" in str(e):
            print("\n⚠️ Phát hiện lỗi: PyTorch không được biên dịch với CUDA")
            print("🔧 Giải pháp: Chạy 'run_cuda_fix.bat' để cài đặt lại PyTorch với CUDA")
        return None

def extract_frames_with_decord(video_path, sample_rate=1, device_id=0):
    """Extract frames from video using Decord library"""
    if not DECORD_AVAILABLE:
        print("⚠️ Decord không khả dụng, sử dụng OpenCV thay thế")
        return extract_frames_with_opencv(video_path, sample_rate)
        
    try:
        # Thử buộc sử dụng GPU nếu được yêu cầu
        force_cuda = os.environ.get('FORCE_CUDA', '0') == '1'
        
        # Kiểm tra GPU thông qua PyTorch
        use_gpu = torch.cuda.is_available() or force_cuda
        
        # Chọn context phù hợp
        if use_gpu:
            print(f"✅ Sử dụng Decord với GPU (device {device_id})")
            try:
                ctx = decord.gpu(device_id)
                decord.bridge.set_bridge('torch')
            except Exception as e:
                print(f"⚠️ Không thể khởi tạo GPU context: {e}")
                print("⚠️ Chuyển sang CPU")
                ctx = decord.cpu()
                decord.bridge.set_bridge('native')
        else:
            print("⚠️ Không tìm thấy GPU hoặc GPU bị tắt, sử dụng CPU")
            ctx = decord.cpu()
            decord.bridge.set_bridge('native')
            
        # Create video reader
        vr = decord.VideoReader(video_path, ctx=ctx)
        fps = vr.get_avg_fps()
        frame_count = len(vr)
        
        # Calculate which frames to extract
        frame_indices = list(range(0, frame_count, int(fps * sample_rate)))
        timestamps = [idx / fps for idx in frame_indices]
        
        # Extract frames
        frames = vr.get_batch(frame_indices).asnumpy()
        
        return frames, timestamps, fps  # Added fps to return values
    except Exception as e:
        print(f"Error using Decord: {e}")
        print("Falling back to standard frame extraction using OpenCV...")
        return None, None, None

def extract_frames_with_opencv(video_path, sample_rate=1):
    """Extract frames using OpenCV as fallback"""
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

def detect_motion_yolo8(video_path, model=None, device="cuda", sample_rate=1, 
                       threshold=0.25, min_track_duration=1.0, class_focus=[0]):
    """
    Detect motion using YOLOv8 with integrated tracking
    
    Args:
        video_path: Path to video file
        model: Pre-initialized YOLOv8 model (detection or pose)
        device: 'cuda' or 'cpu'
        sample_rate: Frames per second to process
        threshold: Detection confidence threshold
        min_track_duration: Minimum duration for a track to be considered
        class_focus: List of class IDs to focus on (default: [0]=person)
    
    Returns:
        List of motion segments [start_time, end_time]
    """
    print("🎬 Đang phân tích và trích xuất các cảnh có chuyển động nhân vật...")
    
    # Initialize model if not provided
    if model is None:
        model = initialize_yolo8_model(device=device)
    
    # Extract frames
    use_gpu = (device == "cuda" and torch.cuda.is_available())
    if DECORD_AVAILABLE:
        if use_gpu:
            print("Sử dụng GPU với Decord để trích xuất frame")
            frames, timestamps, fps = extract_frames_with_decord(video_path, sample_rate=sample_rate, device_id=0)
        else:
            print("Sử dụng CPU với Decord để trích xuất frame")
            frames, timestamps, fps = extract_frames_with_decord(video_path, sample_rate=sample_rate, device_id=-1)
    else:
        frames, timestamps, fps = extract_frames_with_opencv(video_path, sample_rate=sample_rate)
    
    if frames is None or len(frames) == 0:
        print("⚠️ Không thể trích xuất frame từ video")
        return []
    
    print(f"Đang phân tích video: {len(timestamps)} frames, {fps} fps")
    
    # Track objects across frames
    try:
        results = model.track(frames, persist=True, conf=threshold, verbose=False)
    except Exception as e:
        print(f"❌ Lỗi khi tracking: {e}")
        # Fallback to regular detection
        results = [model(frame) for frame in tqdm(frames, desc="Analyzing frames")]
    
    # Analyze detection results
    tracks = {}
    motion_scores = np.zeros(len(frames))
    
    for i, result in enumerate(results):
        if hasattr(result, 'boxes'):
            boxes = result.boxes
            
            # Check if we have tracking IDs
            if hasattr(boxes, 'id') and boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)
                classes = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                
                # Process each detected object
                for j, (track_id, cls, conf) in enumerate(zip(track_ids, classes, confs)):
                    # Only consider specified classes with sufficient confidence
                    if cls in class_focus and conf > threshold:
                        if track_id not in tracks:
                            tracks[track_id] = []
                        tracks[track_id].append((i, timestamps[i]))
                        
                        # Add to motion score
                        motion_scores[i] += conf
            else:
                # No tracking IDs available, fallback to detection counts
                classes = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                
                # Count objects of interest
                for cls, conf in zip(classes, confs):
                    if cls in class_focus and conf > threshold:
                        motion_scores[i] += conf
    
    # Smooth motion scores
    window_size = max(3, int(fps * sample_rate // 2))
    smoothed_scores = np.convolve(motion_scores, np.ones(window_size)/window_size, mode='same')
    
    # Calculate adaptive threshold
    mean_score = np.mean(smoothed_scores)
    std_score = np.std(smoothed_scores)
    threshold = mean_score + std_score * 0.8
    print(f"Phân tích điểm chuyển động: mean={mean_score:.4f}, std={std_score:.4f}")
    print(f"Ngưỡng phát hiện chuyển động: {threshold:.4f}")
    
    # Identify segments with high motion
    motion_segments = []
    in_motion = False
    start_idx = 0
    
    for i, score in enumerate(smoothed_scores):
        if score > threshold and not in_motion:
            in_motion = True
            start_idx = i
        elif (score <= threshold * 0.7 or i == len(smoothed_scores) - 1) and in_motion:
            in_motion = False
            # Convert to time
            if i - start_idx >= (fps * sample_rate * min_track_duration):
                motion_segments.append([timestamps[start_idx], timestamps[i]])
    
    # Add final segment if still in motion
    if in_motion and len(timestamps) - start_idx >= (fps * sample_rate * min_track_duration):
        motion_segments.append([timestamps[start_idx], timestamps[-1]])
    
    # Merge close segments
    if len(motion_segments) > 1:
        merged = [motion_segments[0]]
        
        for current in motion_segments[1:]:
            previous = merged[-1]
            # If gap is less than 2 seconds
            if current[0] - previous[1] < 2.0:
                merged[-1][1] = current[1]  # Extend previous segment
            else:
                merged.append(current)
        
        motion_segments = merged
    
    print(f"✅ Tìm thấy {len(motion_segments)} đoạn có chuyển động nhân vật mạnh")
    
    # Calculate total duration of motion segments
    if motion_segments:
        total_duration = sum(end - start for start, end in motion_segments)
        video_duration = VideoFileClip(video_path).duration
        print(f"Tổng thời lượng chuyển động: {total_duration:.1f}s ({total_duration/video_duration*100:.1f}% của video)")
        
        for i, (start, end) in enumerate(motion_segments):
            print(f"  Đoạn {i+1}: {start:.1f}s - {end:.1f}s (thời lượng: {end-start:.1f}s)")
    
    return motion_segments

def detect_action_with_pose(video_path, model=None, device="cuda", sample_rate=1,
                           min_keypoints=5, movement_threshold=0.3):
    """
    Detect action scenes using YOLOv8 pose estimation
    
    This function tracks human poses and measures their movement over time
    to identify segments with significant human activity
    
    Args:
        video_path: Path to video file
        model: YOLOv8 pose model (if None, will initialize)
        device: 'cuda' or 'cpu'
        sample_rate: Frames per second to process
        min_keypoints: Minimum number of keypoints to consider a valid pose
        movement_threshold: Threshold for considering significant movement
        
    Returns:
        List of action segments [start_time, end_time]
    """
    print("🔍 Phát hiện chuyển động nâng cao với Optical Flow...")
    
    # Initialize pose model if not provided
    if model is None:
        model = initialize_yolo8_model(device=device, model_size="n", task="pose")
    
    # Extract frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"Đang phân tích video: {duration:.1f} giây, {fps} fps")
    
    # Calculate frame indices to process
    frame_skip = int(fps) // 2  # 2 frames per second
    print(f"Lấy mẫu 2 frame/giây (mỗi {frame_skip} frame)")
    frame_indices = list(range(0, frame_count, frame_skip))
    
    # Process frames with pose estimation
    pose_data = []
    timestamps = []
    
    for frame_idx in tqdm(frame_indices, desc="Phân tích optical flow"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Run pose estimation
        results = model(frame, verbose=False)
        timestamp = frame_idx / fps
        timestamps.append(timestamp)
        
        # Extract pose keypoints
        if len(results) > 0 and hasattr(results[0], 'keypoints'):
            keypoints = results[0].keypoints.xy.cpu().numpy()
            if len(keypoints) > 0:
                pose_data.append((timestamp, keypoints))
            else:
                pose_data.append((timestamp, None))
        else:
            pose_data.append((timestamp, None))
    
    cap.release()
    
    # Analyze pose movement over time
    motion_scores = []
    prev_pose = None
    
    for i, (timestamp, keypoints) in enumerate(pose_data):
        if keypoints is None or (prev_pose is None and i > 0):
            motion_scores.append(0.0)
            continue
            
        if prev_pose is None:
            motion_scores.append(0.0)
            prev_pose = keypoints
            continue
        
        # Calculate movement between consecutive poses
        movement_score = 0.0
        valid_keypoints = 0
        
        for person_idx, person_kpts in enumerate(keypoints):
            # Find closest person in previous frame
            if len(prev_pose) > 0:
                prev_person = min(prev_pose, key=lambda p: np.sum((p.mean(axis=0) - person_kpts.mean(axis=0))**2))
                
                # Calculate movement per keypoint
                for kpt_idx, kpt in enumerate(person_kpts):
                    if kpt_idx < len(prev_person):
                        dist = np.sqrt(np.sum((kpt - prev_person[kpt_idx])**2))
                        if dist < 100:  # Ignore large jumps (likely tracking errors)
                            movement_score += dist
                            valid_keypoints += 1
        
        # Normalize by number of keypoints
        if valid_keypoints >= min_keypoints:
            motion_scores.append(movement_score / valid_keypoints)
        else:
            motion_scores.append(0.0)
            
        prev_pose = keypoints
    
    # Smooth scores
    window_size = 5
    smoothed_scores = np.convolve(motion_scores, np.ones(window_size)/window_size, mode='same')
    
    # Calculate adaptive threshold
    mean_score = np.mean(smoothed_scores)
    std_score = np.std(smoothed_scores)
    threshold = mean_score + std_score * 0.7
    print(f"Phân tích điểm chuyển động: mean={mean_score:.4f}, std={std_score:.4f}")
    print(f"Ngưỡng phát hiện chuyển động: {threshold:.4f}")
    
    # Identify segments with significant movement
    action_segments = []
    in_action = False
    start_idx = 0
    
    for i, score in enumerate(smoothed_scores):
        if score > threshold and not in_action:
            in_action = True
            start_idx = i
        elif (score <= threshold * 0.6 or i == len(smoothed_scores) - 1) and in_action:
            in_action = False
            # Minimum 2 seconds duration
            if i - start_idx >= 4:  # At least 4 frames at 2fps
                action_segments.append([timestamps[start_idx], timestamps[i]])
    
    # Add final segment if still in action
    if in_action and len(timestamps) - start_idx >= 4:
        action_segments.append([timestamps[start_idx], timestamps[-1]])
    
    # Merge close segments
    if len(action_segments) > 1:
        merged = [action_segments[0]]
        
        for current in action_segments[1:]:
            previous = merged[-1]
            # If gap is less than 4 seconds
            if current[0] - previous[1] < 4.0:
                merged[-1][1] = current[1]  # Extend previous segment
            else:
                merged.append(current)
        
        action_segments = merged
    
    print(f"✅ Tìm thấy {len(action_segments)} đoạn có chuyển động nhân vật mạnh")
    
    # Calculate total duration of action segments
    if action_segments:
        total_duration = sum(end - start for start, end in action_segments)
        print(f"Tổng thời lượng chuyển động: {total_duration:.1f}s ({total_duration/duration*100:.1f}% của video)")
        
        for i, (start, end) in enumerate(action_segments):
            print(f"  Đoạn {i+1}: {start:.1f}s - {end:.1f}s (thời lượng: {end-start:.1f}s)")
    
    return action_segments

def create_highlight_video_yolo8(video_path, output_path, model=None, device="cuda", 
                              task="detect", buffer_time=1.0):
    """
    Create a highlight video using YOLOv8 detection or pose estimation
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
        model: YOLOv8 model (if None, will be initialized)
        device: 'cuda' or 'cpu'
        task: 'detect' or 'pose'
        buffer_time: Extra time to add before and after each segment
        
    Returns:
        bool: Success status
    """
    print("🚀 Sử dụng CUDA để tăng tốc xuất video")
    
    # Initialize model if needed
    if model is None:
        if task == "pose":
            model = initialize_yolo8_model(device=device, model_size="n", task="pose")
        else:
            model = initialize_yolo8_model(device=device, task="detect")
    
    # Detect action segments based on task
    if task == "pose":
        segments = detect_action_with_pose(video_path, model, device)
    else:
        segments = detect_motion_yolo8(video_path, model, device)
    
    if not segments:
        print("⚠️ Không tìm thấy đoạn video phù hợp để trích xuất")
        return False
    
    # Add buffer time
    clip = VideoFileClip(video_path)
    duration = clip.duration
    
    for i in range(len(segments)):
        start_time = max(0, segments[i][0] - buffer_time)
        end_time = min(duration, segments[i][1] + buffer_time)
        segments[i] = [start_time, end_time]
    
    # Create output clips
    clips = []
    for i, (start, end) in enumerate(segments):
        print(f"Đoạn {i+1}: {start:.1f}s - {end:.1f}s (độ dài: {end-start:.1f}s)")
        clips.append(clip.subclip(start, end))
    
    # Combine clips
    if clips:
        final_clip = concatenate_videoclips(clips)
        
        try:
            # Try to use CUDA acceleration if available
            if torch.cuda.is_available():
                print("Sử dụng CUDA để tăng tốc xuất video")
                final_clip.write_videofile(output_path, codec="h264_nvenc", audio_codec="aac")
            else:
                final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            
            print(f"✅ Đã tạo video highlight thành công: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Lỗi khi xuất video: {e}")
            # Fallback to standard encoding
            try:
                final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
                print(f"✅ Đã tạo video highlight thành công (không dùng GPU): {output_path}")
                return True
            except Exception as e2:
                print(f"❌ Không thể tạo video: {e2}")
                return False
        finally:
            clip.close()
            final_clip.close()
    else:
        print("❌ Không có đoạn nào để tạo video")
        return False

def create_multi_scene_short(video_path, output_path, model=None, device="cuda",
                           target_duration=30.0, max_scenes=5, buffer_time=1.0):
    """
    Create a multi-scene short video with customizable parameters
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
        model: YOLOv8 model (if None, will be initialized)
        device: 'cuda' or 'cpu'
        target_duration: Target duration in seconds
        max_scenes: Maximum number of scenes to include
        buffer_time: Time buffer around each scene
        
    Returns:
        bool: Success status
    """
    print(f"🎬 Tạo video ngắn từ {max_scenes} cảnh quan trọng (tối đa {target_duration}s)...")
    
    # Initialize model if needed
    if model is None:
        model = initialize_yolo8_model(device=device, task="detect")
    
    # Detect motion segments using object tracking
    segments = detect_motion_yolo8(video_path, model, device)
    
    if not segments:
        print("⚠️ Không tìm thấy đoạn video có chuyển động để trích xuất")
        # Fallback to pose detection
        pose_model = initialize_yolo8_model(device=device, model_size="n", task="pose")
        segments = detect_action_with_pose(video_path, pose_model, device)
        
        if not segments:
            print("❌ Không tìm thấy đoạn video phù hợp để trích xuất")
            return False
    
    # Calculate segment scores (simple: longer segments get higher scores)
    segment_scores = [(end - start, i, [start, end]) for i, (start, end) in enumerate(segments)]
    segment_scores.sort(reverse=True)  # Sort by duration, longest first
    
    # Take only max_scenes segments
    selected_segments = [segment for _, _, segment in segment_scores[:max_scenes]]
    
    # Sort by start time to maintain chronological order
    selected_segments.sort()
    
    # Add buffer time
    clip = VideoFileClip(video_path)
    duration = clip.duration
    
    buffered_segments = []
    for start, end in selected_segments:
        buffered_start = max(0, start - buffer_time)
        buffered_end = min(duration, end + buffer_time)
        
        # Check if new segment overlaps with previous
        if buffered_segments and buffered_start < buffered_segments[-1][1]:
            # Merge with previous segment
            buffered_segments[-1][1] = buffered_end
        else:
            buffered_segments.append([buffered_start, buffered_end])
    
    # Calculate total duration and trim if needed
    total_duration = sum(end - start for start, end in buffered_segments)
    
    if total_duration > target_duration:
        print(f"⚠️ Tổng thời lượng ({total_duration:.1f}s) vượt quá giới hạn ({target_duration}s), cắt bớt...")
        
        # First, trim each segment proportionally
        excess = total_duration - target_duration
        for i in range(len(buffered_segments)):
            seg_duration = buffered_segments[i][1] - buffered_segments[i][0]
            trim_amount = min(excess * seg_duration / total_duration, seg_duration * 0.3)
            
            # Trim from both ends equally
            buffered_segments[i][0] += trim_amount / 2
            buffered_segments[i][1] -= trim_amount / 2
            
        # If still over duration, remove shortest segments
        trimmed_segments = sorted(buffered_segments, key=lambda x: x[1] - x[0], reverse=True)
        current_duration = sum(end - start for start, end in trimmed_segments)
        
        while current_duration > target_duration and len(trimmed_segments) > 1:
            trimmed_segments.pop()
            current_duration = sum(end - start for start, end in trimmed_segments)
        
        buffered_segments = sorted(trimmed_segments)
    
    # Create output clips
    clips = []
    for i, (start, end) in enumerate(buffered_segments):
        print(f"Đoạn {i+1}: {start:.1f}s - {end:.1f}s (độ dài: {end-start:.1f}s)")
        clips.append(clip.subclip(start, end))
    
    # Combine clips
    if clips:
        final_clip = concatenate_videoclips(clips)
        
        try:
            # Try to use CUDA acceleration if available
            if torch.cuda.is_available():
                print("Sử dụng CUDA để tăng tốc xuất video")
                final_clip.write_videofile(output_path, codec="h264_nvenc", audio_codec="aac")
            else:
                final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            
            print(f"✅ Đã tạo video highlight thành công: {output_path}")
            final_clip.close()
            clip.close()
            return True
        except Exception as e:
            print(f"❌ Lỗi khi xuất video: {e}")
            # Fallback to standard encoding
            try:
                final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
                print(f"✅ Đã tạo video highlight thành công (không dùng GPU): {output_path}")
                final_clip.close()
                clip.close()
                return True
            except Exception as e2:
                print(f"❌ Không thể tạo video: {e2}")
                clip.close()
                return False
    else:
        print("❌ Không có đoạn nào để tạo video")
        clip.close()
        return False

def create_single_action_short(video_path, output_path, model=None, device="cuda",
                             target_duration=30.0, buffer_time=1.0):
    """
    Create a short video from the most significant single action scene in a video
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
        model: YOLOv8 model (if None, will be initialized)
        device: 'cuda' or 'cpu'
        target_duration: Target duration in seconds
        buffer_time: Time buffer around the scene
        
    Returns:
        bool: Success status
    """
    print(f"🎬 Tạo video short từ cảnh hành động cao trào (tối đa {target_duration}s)...")
    
    # Initialize model if needed
    if model is None:
        # Try pose model first for better action detection
        try:
            model = initialize_yolo8_model(device=device, model_size="n", task="pose")
            segments = detect_action_with_pose(video_path, model, device)
        except Exception as e:
            print(f"⚠️ Lỗi khi sử dụng pose model: {e}")
            print("Chuyển sang sử dụng detection model...")
            model = initialize_yolo8_model(device=device, task="detect")
            segments = detect_motion_yolo8(video_path, model, device)
    else:
        # Use the provided model
        if hasattr(model, 'task') and model.task == 'pose':
            segments = detect_action_with_pose(video_path, model, device)
        else:
            segments = detect_motion_yolo8(video_path, model, device)
    
    if not segments:
        print("❌ Không tìm thấy đoạn video phù hợp để trích xuất")
        return False
    
    # Find the most significant segment (highest motion or longest duration)
    clip = VideoFileClip(video_path)
    duration = clip.duration
    
    # Calculate segment importance by duration
    segment_scores = [(end - start, [start, end]) for start, end in segments]
    segment_scores.sort(reverse=True)  # Sort by duration
    
    # Select the best segment
    _, best_segment = segment_scores[0]
    start, end = best_segment
    
    # Expand segment to target duration if needed
    current_duration = end - start
    if current_duration < target_duration:
        # Calculate how much to expand
        expand_time = min((target_duration - current_duration) / 2.0, 
                         start, duration - end)  # Limit by video boundaries
        start = max(0, start - expand_time)
        end = min(duration, end + expand_time)
    
    # If segment is too long, trim it
    elif current_duration > target_duration:
        # Keep the center portion
        center = (start + end) / 2
        half_target = target_duration / 2
        start = max(0, center - half_target)
        end = min(duration, center + half_target)
    
    # Add buffer time
    start = max(0, start - buffer_time)
    end = min(duration, end + buffer_time)
    
    print(f"Cắt đoạn video từ {start:.1f}s đến {end:.1f}s (độ dài: {end-start:.1f}s)")
    
    # Create output clip
    try:
        output_clip = clip.subclip(start, end)
        
        # Try to use CUDA acceleration if available
        if torch.cuda.is_available():
            print("Sử dụng CUDA để tăng tốc xuất video")
            output_clip.write_videofile(output_path, codec="h264_nvenc", audio_codec="aac")
        else:
            output_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        print(f"✅ Đã tạo video short thành công: {output_path}")
        output_clip.close()
        clip.close()
        return True
    except Exception as e:
        print(f"❌ Lỗi khi xuất video: {e}")
        # Fallback to standard encoding
        try:
            output_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            print(f"✅ Đã tạo video short thành công (không dùng GPU): {output_path}")
            output_clip.close()
            clip.close()
            return True
        except Exception as e2:
            print(f"❌ Không thể tạo video: {e2}")
            clip.close()
            return False

def process_yolo8_short_mode(input_path, output_dir, device="cuda", mode=1, 
                           target_duration=30.0, buffer_time=1.0):
    """
    Process video for short mode using YOLOv8
    
    Args:
        input_path: Path to input video
        output_dir: Output directory
        device: 'cuda' or 'cpu'
        mode: 1 for single action, 2 for multi-scene
        target_duration: Target duration in seconds
        buffer_time: Time buffer around scenes
        
    Returns:
        bool: Success status
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output path
    if mode == 1:
        output_path = os.path.join(output_dir, "short_single_action.mp4")
    else:
        output_path = os.path.join(output_dir, "short_multi_action.mp4")
    
    # Initialize YOLOv8 model
    print("🚀 Tải model YOLOv8 để phát hiện hành động...")
    
    try:
        # First try to initialize pose model for better action detection
        model = initialize_yolo8_model(device=device, model_size="n", task="pose")
        
        # For multi-scene, we also need a detection model for better object tracking
        if mode == 2:
            detect_model = initialize_yolo8_model(device=device, task="detect")
        else:
            detect_model = None
    except Exception as e:
        print(f"⚠️ Không thể khởi tạo pose model: {e}")
        print("Sử dụng detection model thay thế...")
        model = initialize_yolo8_model(device=device, task="detect")
        detect_model = None
    
    # Process according to mode
    if mode == 1:
        print("🎬 Chế độ Single Action - Tạo video từ cảnh hành động cao trào")
        return create_single_action_short(input_path, output_path, model, 
                                        device, target_duration, buffer_time)
    else:
        print("🎬 Chế độ Multi Action - Tạo video từ nhiều cảnh hành động")
        return create_multi_scene_short(input_path, output_path, 
                                      detect_model if detect_model else model,
                                      device, target_duration, max_scenes=5, 
                                      buffer_time=buffer_time)

def fast_extract_segments_cuda(input_path, output_path, segments):
    """Extract and concatenate video segments using CUDA acceleration when possible"""
    if not segments:
        print("Không có đoạn nào để trích xuất")
        return False
        
    try:
        # Tạo thư mục tạm
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_segments")
        os.makedirs(temp_dir, exist_ok=True)
        
        segment_files = []
        for i, (start, end) in enumerate(segments):
            segment_file = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
            duration = end - start
            
            # Sử dụng FFmpeg với hỗ trợ CUDA
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", input_path,
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", 
                segment_file
            ]
            
            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                
                if os.path.exists(segment_file) and os.path.getsize(segment_file) > 0:
                    segment_files.append(segment_file)
                else:
                    print(f"⚠️ Tạo segment {i} không thành công, thử phương pháp thay thế")
            except subprocess.CalledProcessError:
                print(f"⚠️ Lỗi khi tạo segment {i}, thử phương pháp thay thế")
                
        if not segment_files:
            print("❌ Không thể trích xuất đoạn video nào với CUDA")
            print("Chuyển sang phương pháp thay thế...")
            return fast_extract_segments_with_moviepy(input_path, output_path, segments)
            
        # Tạo file danh sách để ghép - FIX: Use forward slashes in paths
        list_file = os.path.join(temp_dir, "segments.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for file in segment_files:
                # Convert backslashes to forward slashes for FFmpeg
                file_path = os.path.basename(file).replace('\\', '/')
                f.write(f"file '{file_path}'\n")
                
        # Ghép các đoạn với FFmpeg
        try:
            # Use direct method with file inputs instead of concat
            if len(segment_files) == 1:
                shutil.copy(segment_files[0], output_path)
                print(f"✅ Chỉ có một đoạn, đã copy trực tiếp thành công: {output_path}")
            else:
                # Use alternative concatenation method to avoid segments.txt issues
                input_args = []
                for file in segment_files:
                    input_args.extend(["-i", file])
                
                filter_complex = ""
                for i in range(len(segment_files)):
                    filter_complex += f"[{i}:v][{i}:a]"
                filter_complex += f"concat=n={len(segment_files)}:v=1:a=1[outv][outa]"
                
                cmd = [
                    "ffmpeg", "-y",
                    *input_args,
                    "-filter_complex", filter_complex,
                    "-map", "[outv]",
                    "-map", "[outa]",
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    output_path
                ]
                
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                
                print(f"✅ Đã tạo video thành công với phương pháp ghép trực tiếp: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Lỗi khi ghép video: {e}")
            print("Chuyển sang phương pháp thay thế với MoviePy...")
            return fast_extract_segments_with_moviepy(input_path, output_path, segments)
        
        # Dọn dẹp
        for file in segment_files:
            try:
                os.remove(file)
            except:
                pass
        
        try:
            if os.path.exists(list_file):
                os.remove(list_file)
            os.rmdir(temp_dir)
        except:
            pass
            
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✅ Đã tạo video thành công: {output_path}")
            return True
        else:
            print("❌ Tạo video thất bại với CUDA")
            return fast_extract_segments_with_moviepy(input_path, output_path, segments)
            
    except Exception as e:
        print(f"❌ Lỗi khi trích xuất video với CUDA: {e}")
        
        # Thử lại với phương pháp thông thường
        print("Chuyển sang phương pháp không dùng CUDA...")
        return fast_extract_segments_with_moviepy(input_path, output_path, segments)

def fast_extract_segments_with_moviepy(input_path, output_path, segments):
    """Fallback method using MoviePy to extract segments"""
    try:
        print("Sử dụng MoviePy để xử lý video...")
        if not segments:
            print("Không có đoạn nào để trích xuất")
            return False
            
        clip = VideoFileClip(input_path)
        clips = []
        
        for start, end in segments:
            start = max(0, start)
            end = min(clip.duration, end)
            if end - start >= 0.5:  # Chỉ lấy đoạn dài ít nhất 0.5 giây
                clips.append(clip.subclip(start, end))
        
        if not clips:
            print("Không có đoạn nào hợp lệ để ghép")
            clip.close()
            return False
            
        print(f"Đang ghép {len(clips)} đoạn video với MoviePy...")
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", 
                                  threads=4, logger=None)
        clip.close()
        final_clip.close()
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✅ Đã tạo video thành công với MoviePy: {output_path}")
            return True
        else:
            print("❌ Tạo video thất bại")
            return False
    except Exception as e:
        print(f"❌ Lỗi khi trích xuất đoạn video với MoviePy: {e}")
        return False

def detect_high_action_scenes_yolo8(video_path, model=None, device="cuda", top_k=5, min_duration=2.0, 
                                  merge_threshold=3.0, confidence=0.25, task="detect"):
    """
    Enhanced detection of high action scenes using YOLOv8 and optical flow
    
    Args:
        video_path: Path to the video file
        model: YOLOv8 model (if None, will initialize one)
        device: Device to run the model on
        top_k: Number of top action scenes to return
        min_duration: Minimum duration for action scenes
        merge_threshold: Threshold for merging nearby scenes
        confidence: Confidence threshold for YOLOv8 detection
        task: Type of YOLOv8 task ("detect", "pose")
        
    Returns:
        List of top action segments as [start_time, end_time]
    """
    print("🎬 Detecting high action scenes with YOLOv8...")
    
    # Initialize YOLOv8 if not provided
    if model is None:
        if task == "pose":
            model = initialize_yolo8_model(device, "yolov8s-pose.pt", task)
        else:
            model = initialize_yolo8_model(device, "yolov8s.pt", task)
            
        if model is None:
            print("⚠️ Failed to initialize YOLOv8, falling back to optical flow only")
            # Fallback to older method if YOLOv8 initialization fails
            # Fallback to simpler detection method
            return detect_motion_yolo8(video_path, None, device, threshold=confidence, min_track_duration=min_duration)
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    
    # Extract frames with Decord for faster processing
    sample_rate = 1  # 1 frame per second
    frames, timestamps = None, None
    
    if DECORD_AVAILABLE:
        try:
            # FIX: Properly handle 3 return values
            frames, timestamps, _ = extract_frames_with_decord(
                video_path, sample_rate=sample_rate, device_id=0 if device == "cuda" else -1)
        except Exception as e:
            print(f"Error with Decord frame extraction: {e}")
            frames, timestamps = None, None
    
    # Fallback to OpenCV if Decord fails
    if frames is None:
        frames, timestamps = extract_frames_with_opencv(video_path, sample_rate=sample_rate)
    
    if not frames:
        print("❌ Failed to extract frames")
        return []
    
    print(f"✅ Extracted {len(frames)} frames for analysis")
    
    # ...existing code...
