import os
import cv2
import numpy as np
import torch
import subprocess
import shutil
from tqdm import tqdm
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Add new imports for optimized video processing
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
    print("⚠️ Decord library not found. Using slower OpenCV for frame extraction. Install with: pip install decord")

def detect_scene_changes(video_path, threshold=30, min_scene_duration=1.0):
    """Phát hiện sự thay đổi cảnh trong video dựa trên sự khác biệt giữa các frame"""
    print(f"Phân tích sự thay đổi cảnh trong video {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở file video: {video_path}")
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    prev_frame = None
    scene_changes = []
    min_frames_between_scenes = int(fps * min_scene_duration)
    last_scene_frame = -min_frames_between_scenes
    
    # Skip frames to speed up processing
    frame_skip = max(1, int(fps / 4))  # Process ~4 frames per second
    
    for frame_idx in tqdm(range(0, total_frames, frame_skip), desc="Phát hiện thay đổi cảnh"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Convert to grayscale and apply blur to reduce noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if prev_frame is None:
            prev_frame = gray
            continue
            
        # Calculate difference between frames
        frame_diff = cv2.absdiff(prev_frame, gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of changed pixels
        change_percent = np.count_nonzero(thresh) * 100 / thresh.size
        
        # Detect significant changes
        if change_percent > threshold and (frame_idx - last_scene_frame) > min_frames_between_scenes:
            time_sec = frame_idx / fps
            scene_changes.append(time_sec)
            last_scene_frame = frame_idx
            
        prev_frame = gray
        
    cap.release()
    print(f"Tìm thấy {len(scene_changes)} thay đổi cảnh.")
    return scene_changes

def detect_motion_activity(video_path, window_size=15, activity_threshold=0.2):
    """Phát hiện các đoạn có nhiều chuyển động trong video"""
    print(f"Phân tích các đoạn có chuyển động mạnh trong video {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở file video: {video_path}")
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    prev_frame = None
    motion_scores = []
    
    # Skip frames to speed up processing
    frame_skip = max(1, int(fps / 4))  # Process ~4 frames per second
    
    for frame_idx in tqdm(range(0, total_frames, frame_skip), desc="Phân tích chuyển động"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None:
            prev_frame = gray
            motion_scores.append(0)
            continue
            
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate motion magnitude
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_score = np.mean(magnitude)
        motion_scores.append(motion_score)
        
        prev_frame = gray
        
    cap.release()
    
    # Apply moving average to smooth motion scores
    smoothed_scores = np.convolve(motion_scores, np.ones(window_size)/window_size, mode='same')
    
    # Find segments with high motion
    high_motion_segments = []
    in_high_motion = False
    start_time = 0
    
    for i, score in enumerate(smoothed_scores):
        time_sec = i * frame_skip / fps
        
        if score > activity_threshold and not in_high_motion:
            in_high_motion = True
            start_time = time_sec
        elif score <= activity_threshold and in_high_motion:
            in_high_motion = False
            high_motion_segments.append((start_time, time_sec))
    
    # Add the last segment if needed
    if in_high_motion:
        high_motion_segments.append((start_time, total_frames / fps))
    
    # Merge segments that are close to each other
    merged_segments = []
    merge_threshold = 2.0  # seconds
    
    for segment in high_motion_segments:
        if not merged_segments or segment[0] - merged_segments[-1][1] > merge_threshold:
            merged_segments.append(list(segment))
        else:
            merged_segments[-1][1] = segment[1]
    
    print(f"Tìm thấy {len(merged_segments)} đoạn có chuyển động mạnh.")
    return merged_segments

def detect_climax_scenes(video_path, threshold_multiplier=1.5, window_size=30):
    """Phát hiện các cảnh cao trào dựa trên chuyển động và thay đổi cảnh"""
    print("Đang phân tích video để tìm cảnh cao trào...")
    
    # Phát hiện chuyển động
    motion_segments = detect_motion_activity(video_path, window_size=window_size, 
                                           activity_threshold=0.3)
    
    # Phát hiện thay đổi cảnh
    scene_changes = detect_scene_changes(video_path, threshold=35)
    
    # Kết hợp thông tin để tìm cảnh cao trào
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Tạo mảng điểm cao trào theo thời gian
    climax_score = np.zeros(int(total_frames / fps) + 1)
    
    # Thêm điểm cho đoạn có chuyển động mạnh
    for start, end in motion_segments:
        start_idx = int(start)
        end_idx = int(end) + 1
        climax_score[start_idx:end_idx] += 1
    
    # Thêm điểm cho thời điểm có thay đổi cảnh
    for time in scene_changes:
        idx = int(time)
        if idx < len(climax_score):
            # Thêm điểm vào khoảng 5 giây xung quanh thay đổi cảnh
            start_idx = max(0, idx - 2)
            end_idx = min(len(climax_score), idx + 3)
            climax_score[start_idx:end_idx] += 0.5
    
    # Làm mịn điểm bằng moving average
    window = np.ones(window_size) / window_size
    smoothed_score = np.convolve(climax_score, window, mode='same')
    
    # Tìm ngưỡng để xác định cao trào
    mean_score = np.mean(smoothed_score)
    threshold = mean_score * threshold_multiplier
    
    # Xác định các đoạn cao trào
    climax_segments = []
    in_climax = False
    start_time = 0
    
    for i, score in enumerate(smoothed_score):
        if score > threshold and not in_climax:
            in_climax = True
            start_time = i
        elif (score <= threshold or i == len(smoothed_score) - 1) and in_climax:
            in_climax = False
            # Chỉ lấy các đoạn đủ dài (ít nhất 3 giây)
            if i - start_time >= 3:
                climax_segments.append([start_time, i])
    
    print(f"Tìm thấy {len(climax_segments)} cảnh cao trào.")
    return climax_segments

def extract_climax_scenes(video_path, output_path, buffer_time=2.0):
    """Trích xuất các cảnh cao trào từ video"""
    climax_segments = detect_climax_scenes(video_path)
    
    if not climax_segments:
        print("Không tìm thấy cảnh cao trào nào.")
        return False
    
    # Thêm buffer time
    for segment in climax_segments:
        segment[0] = max(0, segment[0] - buffer_time)
        segment[1] += buffer_time
    
    # Ghép các đoạn video
    clip = VideoFileClip(video_path)
    clips = [clip.subclip(start, end) for start, end in climax_segments]
    
    if clips:
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_path, codec="libx264")
        print(f"✅ Đã xuất các cảnh cao trào vào: {output_path}")
        return True
    else:
        print("❌ Không thể tạo video cảnh cao trào.")
        return False

def detect_motion_with_object_tracking(video_path, model, device, window_size=15, activity_threshold=0.2):
    """
    Phát hiện các đoạn có nhiều chuyển động của nhân vật chính trong video
    Kết hợp object detection và optical flow để tránh nhận nhầm chuyển động máy quay
    """
    print(f"Phân tích chuyển động của nhân vật chính trong video {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở file video: {video_path}")
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    prev_frame = None
    motion_scores = []
    object_motion_scores = []  # Điểm chuyển động của đối tượng
    camera_motion_scores = []  # Điểm chuyển động của camera
    
    # Chuẩn bị mô hình object detection nếu không truyền vào
    if model is None:
        try:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"Không thể tải mô hình YOLOv5: {e}")
            print("Sử dụng chỉ optical flow mà không có object detection")
            model = None
    
    # Skip frames to speed up processing
    frame_skip = max(1, int(fps / 4))  # Process ~4 frames per second
    
    # Lưu trữ vị trí đối tượng ở frame trước
    prev_objects = []
    
    for frame_idx in tqdm(range(0, total_frames, frame_skip), desc="Phân tích chuyển động"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Convert to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện đối tượng (người) trong frame hiện tại
        current_objects = []
        if model is not None:
            try:
                # Resize frame for model
                frame_resized = cv2.resize(frame, (640, 640))
                
                # Run object detection
                results = model(frame_resized)
                detections = results.xyxy[0].cpu().numpy()
                
                # Filter for persons (class 0)
                for detection in detections:
                    if int(detection[5]) == 0 and float(detection[4]) > 0.3:
                        # Convert bounding box to original frame size
                        x1, y1, x2, y2 = detection[:4]
                        h, w = frame.shape[:2]
                        
                        # Calculate bbox center for tracking movement
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        current_objects.append((center_x, center_y, x2-x1, y2-y1))
            except Exception as e:
                print(f"Lỗi khi phát hiện đối tượng: {e}")
        
        if prev_frame is None:
            prev_frame = gray
            prev_objects = current_objects
            motion_scores.append(0)
            object_motion_scores.append(0)
            camera_motion_scores.append(0)
            continue
            
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate overall motion magnitude
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        overall_motion = np.mean(magnitude)
        motion_scores.append(overall_motion)
        
        # Tách biệt chuyển động đối tượng và chuyển động camera
        object_motion = 0
        camera_motion = overall_motion  # Giả sử ban đầu tất cả là camera motion
        
        # Nếu có đối tượng được phát hiện, tính toán chuyển động của chúng
        if current_objects:
            # Tính tốc độ di chuyển của các đối tượng
            if prev_objects:
                # Tính khoảng cách trung bình di chuyển của các đối tượng
                total_movement = 0
                matched_objects = 0
                
                # Ghép cặp đối tượng giữa các frame dựa trên IOU hoặc khoảng cách
                for curr_obj in current_objects:
                    best_match = None
                    best_dist = float('inf')
                    
                    for prev_obj in prev_objects:
                        dist = np.sqrt((curr_obj[0] - prev_obj[0])**2 + (curr_obj[1] - prev_obj[1])**2)
                        if dist < best_dist:
                            best_dist = dist
                            best_match = prev_obj
                    
                    if best_match and best_dist < max(frame.shape) * 0.2:  # Ngưỡng hợp lý cho vật thể di chuyển
                        total_movement += best_dist
                        matched_objects += 1
                
                if matched_objects > 0:
                    object_motion = total_movement / matched_objects
                    
                    # Ước tính chuyển động camera bằng cách loại bỏ chuyển động đối tượng
                    # Càng nhiều đối tượng di chuyển, ước tính càng chính xác
                    if matched_objects >= 2:
                        camera_motion = max(0, overall_motion - object_motion * 0.7)
            
        object_motion_scores.append(object_motion)
        camera_motion_scores.append(camera_motion)
        
        prev_frame = gray
        prev_objects = current_objects
        
    cap.release()
    
    # Cải thiện: Tính toán tốc độ thay đổi khung hình
    frame_change_rate = []
    for i in range(1, len(motion_scores)):
        frame_change_rate.append(abs(motion_scores[i] - motion_scores[i-1]))
    
    # Thêm giá trị 0 ở đầu để đảm bảo độ dài vector
    frame_change_rate.insert(0, 0)
    
    # Kết hợp các chỉ số để xác định cảnh cao trào
    combined_scores = []
    for i in range(len(motion_scores)):
        # Công thức kết hợp: 
        # - Coi trọng chuyển động đối tượng
        # - Giảm trọng số cho chuyển động camera
        # - Coi trọng tốc độ thay đổi khung hình (đặc trưng của cảnh cao trào)
        if i < len(object_motion_scores) and i < len(camera_motion_scores) and i < len(frame_change_rate):
            object_weight = 2.0
            camera_penalty = 0.5
            change_rate_weight = 1.5
            
            score = (object_motion_scores[i] * object_weight - 
                    camera_motion_scores[i] * camera_penalty + 
                    frame_change_rate[i] * change_rate_weight)
            
            # Đảm bảo không có giá trị âm
            score = max(0, score)
            combined_scores.append(score)
        else:
            combined_scores.append(0)
    
    # Apply moving average to smooth scores
    smoothed_scores = np.convolve(combined_scores, np.ones(window_size)/window_size, mode='same')
    
    # Find segments with significant motion
    high_motion_segments = []
    in_high_motion = False
    start_time = 0
    
    # Tính ngưỡng thích ứng
    mean_score = np.mean(smoothed_scores)
    std_score = np.std(smoothed_scores)
    adaptive_threshold = mean_score + std_score * 0.8  # Điều chỉnh hệ số này theo độ nhạy mong muốn
    print(f"Ngưỡng phát hiện chuyển động thích ứng: {adaptive_threshold:.4f}")
    
    # Sử dụng ngưỡng thích ứng hoặc ngưỡng tối thiểu
    final_threshold = max(activity_threshold, adaptive_threshold)
    
    for i, score in enumerate(smoothed_scores):
        time_sec = i * frame_skip / fps
        
        if score > final_threshold and not in_high_motion:
            in_high_motion = True
            start_time = time_sec
        elif (score <= final_threshold * 0.7 or i == len(smoothed_scores) - 1) and in_high_motion:
            in_high_motion = False
            # Chỉ lấy các đoạn đủ dài (ít nhất 2 giây)
            if time_sec - start_time >= 2:
                high_motion_segments.append((start_time, time_sec))
    
    # Merge segments that are close to each other
    merged_segments = []
    merge_threshold = 3.0  # seconds
    
    for segment in high_motion_segments:
        if not merged_segments or segment[0] - merged_segments[-1][1] > merge_threshold:
            merged_segments.append(list(segment))
        else:
            merged_segments[-1][1] = segment[1]
    
    print(f"Tìm thấy {len(merged_segments)} đoạn có chuyển động mạnh của nhân vật chính.")
    return merged_segments

def detect_climax_scenes_improved(video_path, model, device, threshold_multiplier=1.3, window_size=30):
    """
    Phát hiện các cảnh cao trào dựa trên nhiều tiêu chí phức hợp:
    1. Chuyển động của nhân vật chính (theo dõi đối tượng)
    2. Tốc độ thay đổi khung hình (quan trọng cho cảnh hành động)
    3. Thay đổi cảnh nhanh (cắt nhanh giữa các cảnh)
    4. Tỷ lệ cảnh (ưu tiên cận cảnh, trung cảnh)
    """
    print("Đang phân tích video để tìm cảnh cao trào...")
    
    # Phát hiện chuyển động kết hợp theo dõi đối tượng
    motion_segments = detect_motion_with_object_tracking(
        video_path, model, device, window_size=window_size, activity_threshold=0.25)
    
    # Phát hiện thay đổi cảnh (giảm threshold để nhạy hơn với thay đổi cảnh)
    scene_changes = detect_scene_changes(video_path, threshold=30, min_scene_duration=0.8)
    
    # Phân tích mật độ thay đổi cảnh
    scene_density = analyze_scene_change_density(scene_changes, window_size=15)
    
    # Kết hợp thông tin để tìm cảnh cao trào
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    # Tạo mảng điểm cao trào theo thời gian
    duration_seconds = int(duration) + 1
    climax_score = np.zeros(duration_seconds)
    
    # Thêm điểm cho đoạn có chuyển động mạnh của nhân vật
    for start, end in motion_segments:
        start_idx = int(start)
        end_idx = min(int(end) + 1, duration_seconds)
        climax_score[start_idx:end_idx] += 2.0  # Trọng số cao hơn
    
    # Thêm điểm cho thời điểm có mật độ thay đổi cảnh cao
    for i, density in enumerate(scene_density):
        if i < duration_seconds:
            # Mật độ thay đổi cảnh cao thường là dấu hiệu của cảnh hành động
            climax_score[i] += density * 1.5
    
    # Làm mịn điểm bằng moving average
    window = np.ones(window_size) / window_size
    smoothed_score = np.convolve(climax_score, window, mode='same')
    
    # Tính ngưỡng thích ứng
    non_zero_scores = smoothed_score[smoothed_score > 0]
    if len(non_zero_scores) > 0:
        mean_score = np.mean(non_zero_scores)
        threshold = mean_score * threshold_multiplier
    else:
        # Fallback nếu không có segment nào có điểm
        mean_score = 0.1
        threshold = 0.15
    
    print(f"Điểm trung bình: {mean_score:.3f}, Ngưỡng xác định cao trào: {threshold:.3f}")
    
    # Xác định các đoạn cao trào với khả năng chịu đựng tốt hơn
    climax_segments = []
    in_climax = False
    start_time = 0
    low_count = 0  # Đếm số lượng frame liên tiếp dưới ngưỡng
    
    for i, score in enumerate(smoothed_score):
        if score > threshold and not in_climax:
            in_climax = True
            start_time = i
            low_count = 0
        elif score > threshold * 0.7 and in_climax:
            # Reset bộ đếm nếu điểm vẫn cao
            low_count = 0
        elif score <= threshold * 0.7 and in_climax:
            low_count += 1
            # Chỉ kết thúc đoạn cao trào nếu điểm thấp liên tục
            if low_count >= 5 or i == len(smoothed_score) - 1:
                in_climax = False
                # Chỉ lấy các đoạn đủ dài (ít nhất 3 giây)
                if i - start_time >= 3:
                    climax_segments.append([start_time, i-low_count+1])
                low_count = 0
    
    # Thêm đoạn cao trào cuối cùng nếu kết thúc video trong trạng thái cao trào
    if in_climax and duration_seconds - start_time >= 3:
        climax_segments.append([start_time, duration_seconds-1])
    
    print(f"Tìm thấy {len(climax_segments)} cảnh cao trào.")
    return climax_segments

def analyze_scene_change_density(scene_changes, window_size=15):
    """Phân tích mật độ thay đổi cảnh theo thời gian"""
    if not scene_changes:
        return []
        
    max_time = int(scene_changes[-1]) + window_size
    density = np.zeros(max_time)
    
    # Tạo hàm mật độ Gaussian cho từng thay đổi cảnh
    for change_time in scene_changes:
        time_int = int(change_time)
        # Thêm giá trị Gaussian xung quanh điểm thay đổi cảnh
        for i in range(max(0, time_int - window_size//2), min(max_time, time_int + window_size//2 + 1)):
            # Gaussian weight based on distance
            weight = np.exp(-0.5 * ((i - time_int) / (window_size/5))**2)
            density[i] += weight
    
    # Normalize
    if np.max(density) > 0:
        density = density / np.max(density)
        
    return density

def extract_climax_scenes_improved(video_path, output_path, model, device, buffer_time=2.0, threshold_multiplier=1.0):
    climax_segments = detect_climax_scenes_improved(video_path, model, device, threshold_multiplier)
    
    if not climax_segments:
        print("Không tìm thấy cảnh cao trào")
        return False, []
    
    # Thêm buffer time và đảm bảo thời lượng hợp lý
    clip = VideoFileClip(video_path)
    duration = clip.duration
    
    for i in range(len(climax_segments)):
        start, end = climax_segments[i]
        start = max(0, start - buffer_time)
        end = min(duration, end + buffer_time)
        climax_segments[i] = [start, end]
    
    # Ghép các đoạn video
    clips = []
    for i, (start, end) in enumerate(climax_segments):
        print(f"Cảnh cao trào {i+1}: {start:.1f}s - {end:.1f}s")
        clips.append(clip.subclip(start, end))
    
    if clips:
        print(f"Tạo video với {len(clips)} cảnh cao trào...")
        final_video = concatenate_videoclips(clips)
        final_video.write_videofile(output_path, codec="libx264")
        final_video.close()
        clip.close()
        
        return True, climax_segments
    else:
        print("Không thể tạo video từ các cảnh cao trào")
        clip.close()
        return False, []

def detect_scene_changes_optimized(video_path, threshold=25):
    """Phiên bản tối ưu của phát hiện thay đổi cảnh"""
    print(f"Phát hiện thay đổi cảnh (phiên bản tối ưu)...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở file video: {video_path}")
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    prev_frame = None
    scene_changes = []
    
    # Tăng frame_skip để chỉ xử lý 1 frame mỗi giây 
    frame_skip = max(1, int(fps))
    
    for frame_idx in tqdm(range(0, total_frames, frame_skip), desc="Phát hiện thay đổi cảnh"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Giảm kích thước frame để tăng tốc xử lý
        small_frame = cv2.resize(frame, (160, 90))  # Giảm độ phân giải hơn nữa
        
        # Convert to grayscale and apply blur to reduce noise
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None:
            prev_frame = gray
            continue
            
        # Calculate difference between frames - không cần threshold
        frame_diff = cv2.absdiff(prev_frame, gray)
        
        # Calculate percentage of changed pixels directly from diff
        change_percent = np.mean(frame_diff) * 100 / 255
        
        # Detect significant changes
        if change_percent > threshold:
            time_sec = frame_idx / fps
            scene_changes.append(time_sec)
            
        prev_frame = gray
        
    cap.release()
    print(f"Tìm thấy {len(scene_changes)} thay đổi cảnh.")
    return scene_changes

def detect_motion_activity_optimized(video_path, window_size=15, activity_threshold=0.2, device="cuda"):
    """
    Phiên bản tối ưu của phát hiện chuyển động sử dụng:
    1. Frame skip cao hơn (1 frame/giây)
    2. Frame Difference thay cho Optical Flow
    3. PyTorch tensor xử lý trên GPU
    """
    print(f"Phân tích chuyển động với phương pháp tối ưu...")
    
    import torch
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở file video: {video_path}")
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Tăng frame_skip để chỉ xử lý 1 frame mỗi giây
    frame_skip = max(1, int(fps))  # Đây là thay đổi chính - xử lý 1 frame/giây
    print(f"Frame skip: {frame_skip} (1 frame mỗi giây)")
    
    # Xác định số lượng frame cần xử lý và tạo progress bar
    num_frames = int(total_frames / frame_skip)
    
    prev_frame = None
    motion_scores = []
    
    # Tạo một tensor rỗng trên device để kiểm tra xem GPU có sẵn sàng không
    try:
        if device == "cuda" and torch.cuda.is_available():
            # Khởi tạo tensor rỗng để xác nhận GPU hoạt động
            test_tensor = torch.zeros((1,), device=device)
            del test_tensor  # Giải phóng ngay
            print("✅ Sử dụng GPU để xử lý hình ảnh")
        else:
            device = "cpu"
            print("⚠️ Không phát hiện được GPU, sử dụng CPU")
    except Exception as e:
        device = "cpu"
        print(f"⚠️ Lỗi khi kiểm tra GPU: {e}, sử dụng CPU")
    
    # Xử lý từng frame
    for frame_idx in tqdm(range(0, total_frames, frame_skip), desc="Phân tích chuyển động"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Giảm kích thước frame để tăng tốc xử lý
        small_frame = cv2.resize(frame, (320, 180))
        
        # Chuyển sang grayscale
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None:
            prev_frame = gray
            motion_scores.append(0)
            continue
        
        try:
            if device == "cuda":
                # Chuyển frame sang PyTorch tensor trên GPU
                prev_tensor = torch.tensor(prev_frame, dtype=torch.float32, device=device)
                curr_tensor = torch.tensor(gray, dtype=torch.float32, device=device)
                
                # Tính frame difference bằng PyTorch
                diff_tensor = torch.abs(curr_tensor - prev_tensor)
                motion_score = torch.mean(diff_tensor).item()
            else:
                # Sử dụng numpy cho CPU processing
                frame_diff = cv2.absdiff(prev_frame, gray)
                motion_score = np.mean(frame_diff)
        except Exception as e:
            print(f"Lỗi khi xử lý frame {frame_idx}: {e}")
            # Fallback to CPU
            frame_diff = cv2.absdiff(prev_frame, gray)
            motion_score = np.mean(frame_diff)
            
        # Điều chỉnh hệ số để motion score tương đương với phương pháp optical flow
        motion_score *= 5.0  # Điều chỉnh hệ số để có thang điểm tương tự
        motion_scores.append(motion_score)
        
        prev_frame = gray
        
        # Giải phóng bộ nhớ GPU định kỳ
        if device == "cuda" and frame_idx % 100 == 0:
            torch.cuda.empty_cache()
    
    cap.release()
    
    # Clean up CUDA memory
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Apply moving average to smooth motion scores
    smoothed_scores = np.convolve(motion_scores, np.ones(window_size)/window_size, mode='same')
    
    # Find segments with high motion
    high_motion_segments = []
    in_high_motion = False
    start_time = 0
    
    for i, score in enumerate(smoothed_scores):
        time_sec = i * frame_skip / fps
        
        if score > activity_threshold and not in_high_motion:
            in_high_motion = True
            start_time = time_sec
        elif score <= activity_threshold and in_high_motion:
            in_high_motion = False
            high_motion_segments.append((start_time, time_sec))
    
    # Add the last segment if needed
    if in_high_motion:
        high_motion_segments.append((start_time, duration))
    
    # Merge segments that are close to each other
    merged_segments = []
    merge_threshold = 3.0  # seconds
    
    for segment in high_motion_segments:
        if not merged_segments or segment[0] - merged_segments[-1][1] > merge_threshold:
            merged_segments.append(list(segment))
        else:
            merged_segments[-1][1] = segment[1]
    
    print(f"Tìm thấy {len(merged_segments)} đoạn có chuyển động mạnh.")
    return merged_segments

def detect_climax_scenes_fast(video_path, model=None, device="cuda", threshold_multiplier=1.0):
    """Phiên bản tối ưu và nhanh hơn của phát hiện cảnh cao trào"""
    print("Đang phân tích video để tìm cảnh cao trào (phiên bản tối ưu)...")
    
    # Kiểm tra có sử dụng GPU không
    use_cuda = device == "cuda" and torch.cuda.is_available()
    if use_cuda:
        print("✅ Sử dụng GPU để tăng tốc phát hiện cảnh cao trào")
    
    # 1. Phát hiện chuyển động với phương pháp tối ưu
    motion_segments = detect_motion_activity_optimized(
        video_path, window_size=15, activity_threshold=0.3, device=device)
    
    # 2. Phát hiện thay đổi cảnh với phương pháp tối ưu
    # Tăng frame_skip để chỉ kiểm tra ít frame hơn
    scene_changes = detect_scene_changes_optimized(video_path, threshold=25)
    
    # Kết hợp thông tin để tìm cảnh cao trào
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    # Chuyển đổi kết quả thành cảnh cao trào
    climax_segments = []
    
    # Ưu tiên các đoạn có chuyển động từ phân tích motion
    if motion_segments:
        print(f"Sử dụng {len(motion_segments)} đoạn chuyển động làm cảnh cao trào")
        climax_segments = [segment for segment in motion_segments]
    
    # Nếu không có đoạn chuyển động, dựa vào scene changes
    elif scene_changes:
        print("Dựa vào scene changes để xác định cảnh cao trào")
        prev_time = 0
        
        for time in scene_changes:
            # Tạo đoạn ngắn (5 giây) xung quanh mỗi scene change
            start_time = max(0, time - 2)
            end_time = min(duration, time + 3)
            
            # Chỉ lấy nếu đoạn đủ dài và không chồng lấp với đoạn trước
            if end_time - start_time >= 2 and start_time > prev_time:
                climax_segments.append([start_time, end_time])
                prev_time = end_time
    
    # Nếu vẫn không tìm thấy, lấy một số đoạn đều trong video
    if not climax_segments:
        print("Không tìm thấy cảnh cao trào, lấy một số đoạn đều trong video")
        segment_duration = min(15, max(duration / 10, 5))
        
        # Lấy tối đa 5 đoạn đều
        for i in range(5):
            start_time = duration * (i + 1) / 6
            end_time = min(duration, start_time + segment_duration)
            if end_time - start_time > 3:
                climax_segments.append([start_time, end_time])
    
    print(f"Tìm thấy {len(climax_segments)} cảnh cao trào.")
    return climax_segments

def extract_climax_scenes_optimized(video_path, output_path, model=None, device="cuda", buffer_time=2.0):
    """Phiên bản tối ưu của trích xuất cảnh cao trào"""
    # Phát hiện cao trào với thuật toán tối ưu
    climax_segments = detect_climax_scenes_fast(video_path, model, device)
    
    if not climax_segments:
        print("Không tìm thấy cảnh cao trào nào.")
        return False
    
    # Thêm buffer time và đảm bảo thời lượng hợp lý
    clip = VideoFileClip(video_path)
    duration = clip.duration
    
    # Process segments
    for segment in climax_segments:
        segment[0] = max(0, segment[0] - buffer_time)
        segment[1] = min(duration, segment[1] + buffer_time)
    
    # Merge segments that are close to each other
    merged_segments = []
    merge_threshold = 4.0  # seconds
    
    for segment in sorted(climax_segments, key=lambda x: x[0]):
        if not merged_segments or segment[0] - merged_segments[-1][1] > merge_threshold:
            merged_segments.append(segment)
        else:
            merged_segments[-1][1] = max(merged_segments[-1][1], segment[1])
    
    # Calculate total extract duration
    total_duration = sum(end - start for start, end in merged_segments)
    percentage = (total_duration / duration) * 100
    print(f"Tổng thời lượng cảnh cao trào: {total_duration:.1f}s ({percentage:.1f}% của video gốc)")
    
    # Extract video segments with CUDA acceleration if available
    if device == "cuda" and torch.cuda.is_available():
        print("🚀 Sử dụng GPU để tăng tốc xuất video")
        return fast_extract_segments_cuda(video_path, output_path, merged_segments)
    else:
        # Extract video segments with standard method
        clips = []
        for i, (start, end) in enumerate(merged_segments):
            print(f"Đoạn cao trào {i+1}/{len(merged_segments)}: {start:.1f}s - {end:.1f}s (độ dài {end-start:.1f}s)")
            try:
                clips.append(clip.subclip(start, end))
            except Exception as e:
                print(f"Lỗi khi cắt đoạn {start}-{end}: {e}")
        
        if clips:
            try:
                final_clip = concatenate_videoclips(clips)
                final_clip.write_videofile(output_path, codec="libx264", threads=4)
                print(f"✅ Đã xuất các cảnh cao trào vào: {output_path}")
                clip.close()
                return True
            except Exception as e:
                print(f"❌ Lỗi khi tạo video cảnh cao trào: {e}")
                clip.close()
                return False
        else:
            print("❌ Không thể tạo video cảnh cao trào.")
            clip.close()
            return False

def extract_segments_to_video(video_path, output_path, segments, buffer_time=0.0):
    """Trích xuất các đoạn thời gian cụ thể từ video"""
    if not segments:
        print("Không có đoạn nào để trích xuất")
        return False
        
    try:
        clip = VideoFileClip(video_path)
        clips = []
        
        for start, end in segments:
            # Thêm buffer_time và đảm bảo thời gian hợp lệ
            start_time = max(0, start - buffer_time)
            end_time = min(clip.duration, end + buffer_time)
            
            if end_time - start_time >= 1.0:  # Chỉ lấy đoạn dài hơn 1 giây
                try:
                    segment_clip = clip.subclip(start_time, end_time)
                    clips.append(segment_clip)
                except Exception as e:
                    print(f"Lỗi khi cắt đoạn {start_time}-{end_time}: {e}")
        
        if clips:
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(output_path, codec="libx264", threads=4)
            final_clip.close()
            clip.close()
            print(f"✅ Đã trích xuất {len(clips)} đoạn thành video: {output_path}")
            return True
        else:
            clip.close()
            print("⚠️ Không có đoạn nào hợp lệ để trích xuất")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi khi trích xuất đoạn video: {e}")
        return False

def fast_motion_detection(video_path, frame_rate=1, device='cuda'):
    """Entry point for fast motion detection - uses Decord if available, OpenCV otherwise"""
    if DECORD_AVAILABLE and (device == 'cuda' or torch.cuda.is_available()):
        print("🚀 Sử dụng Decord để tăng tốc phát hiện chuyển động")
        # Remove the recursive call and implement the decord detection inline
        return _fast_motion_detection_with_decord_impl(video_path, frame_rate, device)
    
    print("⚡ Phân tích chuyển động với phương pháp siêu tốc (OpenCV)...")
    
    # Original implementation as fallback
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    # ...existing code (rest of the OpenCV implementation)...
    
def _fast_motion_detection_with_decord_impl(video_path, frame_rate=1, device="cuda"):
    """Internal implementation for Decord-based motion detection"""
    print("⚡ Analyzing motion with optimized method using Decord...")
    
    # Extract frames using Decord
    device_id = 0 if device == "cuda" else -1
    frames, timestamps = extract_frames_with_decord(video_path, sample_rate=frame_rate, device_id=device_id)
    
    # Fall back to OpenCV if Decord fails
    if frames is None:
        print("Falling back to standard frame extraction using OpenCV...")
        # Directly use OpenCV version without recursion
        return _fast_motion_detection_opencv_impl(video_path, frame_rate)
    
    # Process frames for motion detection
    # ...existing code (rest of the Decord implementation)...

def _fast_motion_detection_opencv_impl(video_path, frame_rate=1):
    """OpenCV implementation of motion detection without any recursion"""
    print("⚡ Phân tích chuyển động với phương pháp siêu tốc (OpenCV)...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Chỉ lấy mẫu 1 frame mỗi frame_rate giây
    frame_skip = int(fps * frame_rate)
    prev_frame = None
    motion_scores = []
    timestamps = []
    
    for frame_idx in tqdm(range(0, total_frames, frame_skip), desc="Phát hiện chuyển động"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Giảm độ phân giải để tăng tốc
        small_frame = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None:
            prev_frame = gray
            motion_scores.append(0)
            timestamps.append(frame_idx / fps)
            continue
        
        # Frame difference thay vì optical flow - nhanh hơn ~10x
        frame_diff = cv2.absdiff(prev_frame, gray)
        motion_score = np.mean(frame_diff) / 255.0 * 10  # Normalize và scale
        motion_scores.append(motion_score)
        timestamps.append(frame_idx / fps)
        
        prev_frame = gray
        
    cap.release()
    
    # Phát hiện các đoạn có chuyển động mạnh
    threshold = np.mean(motion_scores) * 1.5  # Adaptive threshold
    if threshold < 0.3:  # Minimum threshold
        threshold = 0.3
        
    segments = []
    in_motion = False
    start_time = 0
    
    for i, score in enumerate(motion_scores):
        if score > threshold and not in_motion:
            in_motion = True
            start_time = timestamps[i]
        elif score <= threshold * 0.7 and in_motion:
            in_motion = False
            if timestamps[i] - start_time >= 1.5:  # Đoạn dài ít nhất 1.5s
                segments.append([start_time, timestamps[i]])
                
    if in_motion:
        segments.append([start_time, duration])
        
    # Ghép các đoạn gần nhau
    if not segments:
        return []
        
    merged = [segments[0]]
    for current in segments[1:]:
        previous = merged[-1]
        if current[0] - previous[1] < 3.0:  # 3s gap
            previous[1] = current[1]
        else:
            merged.append(current)
            
    print(f"Tìm thấy {len(merged)} đoạn chuyển động")
    return merged

def fast_motion_detection_with_decord(video_path, frame_rate=1, device="cuda"):
    """Optimized motion detection using Decord for frame extraction"""
    # Call the internal implementation directly to avoid recursion
    return _fast_motion_detection_with_decord_impl(video_path, frame_rate, device)
    
def check_cuda_ffmpeg_support():
    """Kiểm tra hỗ trợ CUDA và NVENC trong FFmpeg"""
    try:
        # Check CUDA
        result = subprocess.run(
            ["ffmpeg", "-hwaccels"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        output = result.stdout.decode() + result.stderr.decode()
        cuda_available = "cuda" in output.lower()
        
        # Check NVENC
        result = subprocess.run(
            ["ffmpeg", "-encoders"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        output = result.stdout.decode()
        nvenc_available = "h264_nvenc" in output
        
        return cuda_available, nvenc_available
    except Exception as e:
        print(f"Lỗi khi kiểm tra FFmpeg: {e}")
        return False, False

def extract_video_segment_ffmpeg(input_path, output_path, start_time, duration, use_cuda=True):
    """Trích xuất một đoạn video sử dụng FFmpeg - nhanh hơn MoviePy 2-3x"""
    import subprocess
    
    try:
        # Kiểm tra hỗ trợ CUDA và NVENC
        if use_cuda:
            cuda_available, nvenc_available = check_cuda_ffmpeg_support()
        else:
            cuda_available, nvenc_available = False, False
            
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration)
        ]
        
        # Thêm tùy chọn CUDA nếu khả dụng
        if cuda_available:
            cmd.extend(["-hwaccel", "cuda"])
            
        # Sử dụng NVENC nếu khả dụng, nếu không thì dùng preset ultrafast
        if nvenc_available:
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "hq"])
        else:
            cmd.extend(["-c:v", "libx264", "-preset", "ultrafast"])
            
        cmd.extend(["-c:a", "aac", output_path])
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"Lỗi khi trích xuất video: {e}")
        return False

def fast_extract_segments(input_path, output_path, segments, use_cuda=True):
    """Trích xuất và ghép nhiều đoạn video nhanh chóng với FFmpeg"""
    try:
        if not segments:
            print("Không có đoạn nào để trích xuất")
            return False
            
        clip = VideoFileClip(input_path)
        clips = []
        
        for start, end in segments:
            clips.append(clip.subclip(max(0, start), min(clip.duration, end)))
        
        if not clips:
            print("Không có đoạn nào hợp lệ để ghép")
            return False
            
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_path, codec="libx264")
        clip.close()
        final_clip.close()
        
        return True
    except Exception as e:
        print(f"Lỗi khi trích xuất đoạn video: {e}")
        return False

def extract_frames_with_decord(video_path, sample_rate=1, device_id=0):
    """Extract frames using Decord - much faster than OpenCV"""
    if not DECORD_AVAILABLE:
        print("Decord not available, falling back to OpenCV")
        return None, None, None
    
    try:
        # Use CPU context if device_id is -1
        if device_id < 0:
            print("Sử dụng Decord với CPU context")
            ctx = decord.cpu()
        else:
            # Force GPU usage if available
            if torch.cuda.is_available():
                print(f"Sử dụng Decord với GPU context (device {device_id})")
                try:
                    ctx = decord.gpu(device_id)
                except Exception as e:
                    print(f"Lỗi khi tạo GPU context: {e}")
                    print("Fallback to CPU context")
                    ctx = decord.cpu()
            else:
                print("GPU không khả dụng, sử dụng CPU context")
                ctx = decord.cpu()
        
        vr = decord.VideoReader(video_path, ctx=ctx)
        
        fps = vr.get_avg_fps()
        frame_count = len(vr)
        duration = frame_count / fps
        
        # Calculate frames to extract (1 frame per second)
        frame_indices = list(range(0, frame_count, int(fps * sample_rate)))
        timestamps = [idx / fps for idx in frame_indices]
        
        # Extract the frames in batches to prevent memory issues
        batch_size = 16
        frames = []
        
        for i in tqdm(range(0, len(frame_indices), batch_size), desc="Extracting frames"):
            batch_indices = frame_indices[i:i+batch_size]
            batch_frames = vr.get_batch(batch_indices).asnumpy()
            
            # Convert to BGR (OpenCV format) if needed for consistency with other code
            for frame in batch_frames:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Release memory explicitly
            del batch_frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return frames, timestamps, fps
        
    except Exception as e:
        print(f"Error using Decord: {e}")
        print("Falling back to OpenCV for frame extraction")
        return None, None, None

def extract_video_with_cuda(input_path, output_path, start_time, duration):
    """Extract video segment using CUDA acceleration with FFmpeg"""
    try:
        # Check if FFmpeg is available with CUDA support
        result = subprocess.run(
            ["ffmpeg", "-hwaccels"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        cuda_available = "cuda" in result.stdout.decode().lower() or "cuda" in result.stderr.decode().lower()
        nvenc_available = False
        
        # Check if NVENC is available
        result = subprocess.run(
            ["ffmpeg", "-encoders"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        output = result.stdout.decode()
        if "h264_nvenc" in output:
            nvenc_available = True
        
        cmd = ["ffmpeg", "-y"]
        
        # Add seek before input (faster)
        cmd.extend(["-ss", str(start_time)])
        
        # Add input
        cmd.extend(["-i", input_path])
        
        # Add duration
        cmd.extend(["-t", str(duration)])
        
        # Use hardware acceleration if available
        if cuda_available:
            cmd.extend(["-hwaccel", "cuda"])
            
        # Use NVENC encoder if available
        if nvenc_available:
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "hq"])
        else:
            cmd.extend(["-c:v", "libx264", "-preset", "ultrafast"])
            
        # Audio codec
        cmd.extend(["-c:a", "aac"])
        
        # Output file
        cmd.append(output_path)
        
        # Run FFmpeg
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return os.path.exists(output_path)
    
    except Exception as e:
        print(f"Error using FFmpeg with CUDA: {e}")
        # Fall back to CPU version
        return extract_video_segment_ffmpeg(input_path, output_path, start_time, duration)

def fast_extract_segments_cuda(input_path, output_path, segments):
    """Extract and concatenate video segments using CUDA acceleration when possible"""
    if not segments:
        print("Không có đoạn nào để trích xuất")
        return False
        
    try:
        # Kiểm tra xem có C++ extensions hay không
        from utils.gpu_utils import try_load_cpp_extensions
        cpp_extensions = try_load_cpp_extensions()
        
        # Tạo thư mục tạm cho video concat
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_segments")
        os.makedirs(temp_dir, exist_ok=True)
        
        segment_files = []
        
        # Sử dụng C++ extensions nếu có
        if cpp_extensions and "video_extract" in cpp_extensions:
            print("🚀 Sử dụng C++ extensions để xử lý video")
            for i, (start, end) in enumerate(segments):
                segment_file = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
                duration = end - start
                
                try:
                    # Sử dụng hàm C++ để trích xuất đoạn video
                    success = cpp_extensions["video_extract"].extract_segment(
                        input_path, segment_file, start, duration, True)
                    
                    if success and os.path.exists(segment_file) and os.path.getsize(segment_file) > 0:
                        segment_files.append(segment_file)
                    else:
                        print(f"⚠️ Trích xuất segment {i} với C++ không thành công, dùng FFmpeg")
                        # Fall back to FFmpeg for this segment
                        cmd = [
                            "ffmpeg", "-y",
                            "-ss", str(start),
                            "-i", input_path,
                            "-t", str(duration),
                            "-c:v", "libx264", "-preset", "ultrafast",
                            "-c:a", "aac", 
                            segment_file
                        ]
                        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                        if os.path.exists(segment_file) and os.path.getsize(segment_file) > 0:
                            segment_files.append(segment_file)
                except Exception as e:
                    print(f"⚠️ Lỗi khi sử dụng C++ extensions: {e}")
                    # Fall back to FFmpeg for this segment
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
                    except subprocess.CalledProcessError:
                        print(f"⚠️ Lỗi khi tạo segment {i}, thử phương pháp thay thế")
        else:
            # Sử dụng FFmpeg với hỗ trợ CUDA nếu có thể
            for i, (start, end) in enumerate(segments):
                segment_file = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
                duration = end - start
                
                # Sử dụng FFmpeg để trích xuất đoạn video
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
                except subprocess.CalledProcessError:
                    print(f"⚠️ Lỗi khi tạo segment {i}, thử phương pháp thay thế")
        
        # Kiểm tra xem có segment files nào được tạo không
        if not segment_files:
            print("⚠️ Không thể trích xuất đoạn video nào với CUDA")
            print("Chuyển sang phương pháp thay thế...")
            return fast_extract_segments_with_moviepy(input_path, output_path, segments)
        
        # Ghép các đoạn video
        if len(segment_files) == 1:
            # Nếu chỉ có 1 đoạn, copy trực tiếp
            shutil.copy(segment_files[0], output_path)
            print(f"✅ Đã tạo video thành công với phương pháp trực tiếp: {output_path}")
        else:
            # Sử dụng video_concat_cpp nếu có
            if cpp_extensions and "video_concat" in cpp_extensions:
                try:
                    success = cpp_extensions["video_concat"].concatenate_videos(segment_files, output_path)
                    if success:
                        print(f"✅ Đã tạo video thành công với C++ concat: {output_path}")
                    else:
                        raise Exception("Ghép video không thành công với C++")
                except Exception as e:
                    print(f"⚠️ Lỗi khi ghép video với C++: {e}")
                    print("Chuyển sang phương pháp thay thế với FFmpeg...")
                    # Fall back to FFmpeg concat
                    try:
                        # Chuẩn bị file danh sách - Sử dụng forward slashes cho FFmpeg
                        list_file = os.path.join(temp_dir, "segments.txt")
                        with open(list_file, 'w') as f:
                            for file in segment_files:
                                f.write(f"file '{file.replace(os.sep, '/')}'\n")
                        
                        cmd = [
                            "ffmpeg", "-y",
                            "-f", "concat",
                            "-safe", "0",
                            "-i", list_file,
                            "-c:v", "copy",
                            "-c:a", "copy",
                            output_path
                        ]
                        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                        print(f"✅ Đã tạo video thành công với phương pháp ghép FFmpeg: {output_path}")
                    except subprocess.CalledProcessError as e:
                        print(f"⚠️ Lỗi khi ghép video với FFmpeg: {e}")
                        print("Chuyển sang phương pháp thay thế với MoviePy...")
                        return fast_extract_segments_with_moviepy(input_path, output_path, segments)
            else:
                # Sử dụng FFmpeg để ghép video
                try:
                    # Chuẩn bị file danh sách - Sử dụng forward slashes cho FFmpeg
                    list_file = os.path.join(temp_dir, "segments.txt")
                    with open(list_file, 'w') as f:
                        for file in segment_files:
                            f.write(f"file '{file.replace(os.sep, '/')}'\n")
                    
                    cmd = [
                        "ffmpeg", "-y",
                        "-f", "concat",
                        "-safe", "0",
                        "-i", list_file,
                        "-c:v", "copy",
                        "-c:a", "copy",
                        output_path
                    ]
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    print(f"✅ Đã tạo video thành công với phương pháp ghép FFmpeg: {output_path}")
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

def detect_action_with_yolo_batched(video_path, model=None, device="cuda", batch_size=8, frame_rate=1):
    """
    Phát hiện cảnh hành động sử dụng YOLOv5 với xử lý batch để tối ưu hóa.
    Chỉ tập trung vào phát hiện người và phân tích chuyển động của họ.
    """
    print("🚀 Phát hiện cảnh hành động với YOLOv5 (xử lý batch)...")
    
    # Kiểm tra device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA không khả dụng, sử dụng CPU thay thế")
        device = "cpu"
    
    # Khởi tạo mô hình nếu chưa được truyền vào
    if model is None:
        try:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.to(device)
            # Chỉ phát hiện người (class 0)
            model.classes = [0]  # Chỉ giữ lại class "person"
            print("✅ Đã khởi tạo YOLOv5 model, chỉ nhận diện người")
        except Exception as e:
            print(f"❌ Lỗi khi khởi tạo YOLOv5: {e}")
            return []
    elif hasattr(model, 'classes'):
        # Đảm bảo model đã cho có thể nhận diện người
        model.classes = [0]
        print("Đã cập nhật model để chỉ nhận diện người")
    
    # Extract frames using Decord for optimal performance if available
    if DECORD_AVAILABLE:
        print("Sử dụng Decord để trích xuất frames nhanh hơn")
        frames, timestamps = extract_frames_with_decord(video_path, sample_rate=frame_rate, 
                                                     device_id=0 if device == "cuda" else -1)
    else:
        # Fall back to OpenCV
        print("Sử dụng OpenCV để trích xuất frames")
        frames, timestamps = extract_frames_with_opencv(video_path, sample_rate=frame_rate)
        
    if not frames:
        print("❌ Không thể trích xuất frames từ video")
        return []
        
    print(f"Đã trích xuất {len(frames)} frames từ video")
    
    # Process frames in batches
    action_scores = []
    person_counts = []
    
    for i in tqdm(range(0, len(frames), batch_size), desc="Phân tích hành động"):
        batch = frames[i:i+batch_size]
        
        # Run inference on batch
        with torch.no_grad():
            results = model(batch)
            
        # Process detection results for each frame
        for j, frame_result in enumerate(results.xyxy):
            frame_idx = i + j
            if frame_idx >= len(timestamps):
                continue
            
            # Extract people detections
            people = []
            for det in frame_result:
                if int(det[5]) == 0:  # Class 0 is person
                    bbox = det[:4].cpu().numpy()  # x1, y1, x2, y2
                    conf = float(det[4])
                    if conf > 0.3:  # Only keep confident detections
                        people.append(bbox)
            
            # Calculate action score based on number of people and their positions
            person_count = len(people)
            person_counts.append(person_count)
            
            # More sophisticated action score calculation:
            # - More people = higher score
            # - People spread across the frame = higher score (more action)
            action_score = person_count
            
            if person_count > 1:
                # Calculate spatial distribution of people
                centers = []
                for bbox in people:
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    centers.append((center_x, center_y))
                
                # Calculate average distance between people
                if len(centers) > 1:
                    distances = []
                    for idx1 in range(len(centers)):
                        for idx2 in range(idx1 + 1, len(centers)):
                            c1 = centers[idx1]
                            c2 = centers[idx2]
                            dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                            distances.append(dist)
                    avg_distance = np.mean(distances) if distances else 0
                    
                    # Normalize by image size
                    h, w = frames[0].shape[:2]
                    max_dist = np.sqrt(h**2 + w**2)
                    normalized_dist = avg_distance / max_dist
                    
                    # Add distance factor to score (0.5-1.5x multiplier)
                    action_score *= (0.5 + normalized_dist)
            
            action_scores.append(action_score)
    
    # Create segments with high action scores
    if not action_scores:
        return []
    
    # Normalize scores
    max_score = max(action_scores) if action_scores else 1
    normalized_scores = [score / max_score for score in action_scores]
    
    # Calculate adaptive threshold
    mean_score = np.mean(normalized_scores)
    std_score = np.std(normalized_scores)
    threshold = mean_score + 0.5 * std_score  # Adaptive threshold
    
    # Find segments with high action
    high_action_segments = []
    in_segment = False
    start_idx = 0
    
    for i, score in enumerate(normalized_scores):
        if score > threshold and not in_segment:
            in_segment = True
            start_idx = i
        elif (score <= threshold * 0.7 or i == len(normalized_scores) - 1) and in_segment:
            in_segment = False
            # Chỉ lấy các đoạn đủ dài (ít nhất 2 frames)
            if i - start_idx >= 2:
                start_time = timestamps[start_idx]
                end_time = timestamps[i]
                high_action_segments.append([start_time, end_time])
    
    # Cleanup
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Merge nearby segments
    if high_action_segments:
        merged_segments = [high_action_segments[0]]
        merge_threshold = 3.0  # seconds
        
        for segment in high_action_segments[1:]:
            if segment[0] - merged_segments[-1][1] < merge_threshold:
                # Merge with previous segment
                merged_segments[-1][1] = segment[1]
            else:
                merged_segments.append(segment)
                
    return merged_segments

def extract_frames_with_opencv(video_path, sample_rate=1):
    """Extract frames using OpenCV at specified sample rate"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video file: {video_path}")
        return [], [], []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extract one frame per second or at specified rate
    frame_interval = int(fps / sample_rate)
    frames = []
    timestamps = []
    
    for frame_idx in tqdm(range(0, total_frames, frame_interval), desc="Trích xuất frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frames.append(frame)
        timestamps.append(frame_idx / fps)
        
    cap.release()
    return frames, timestamps

def extract_top_action_scenes(video_path, output_path, model=None, device="cuda", 
                            max_scenes=5, max_duration=60, buffer_time=1.0):
    """
    Trích xuất các cảnh hành động hấp dẫn nhất từ video
    và tạo một video ngắn chứa những cảnh này.
    """
    print(f"Phân tích video để trích xuất tối đa {max_scenes} cảnh hành động...")
    
    # Phát hiện cảnh hành động
    action_segments = detect_action_with_yolo_batched(video_path, model, device)
    
    if not action_segments:
        print("Không tìm thấy cảnh hành động nào")
        return False
    
    print(f"Tìm thấy {len(action_segments)} cảnh hành động")
    
    # Sort segments by duration (longer = more interesting)
    action_segments.sort(key=lambda x: x[1] - x[0], reverse=True)
    # Merge nearby segments
    # Limit to max_scenes
    selected_segments = action_segments[:max_scenes]
    # Apply buffer time
    clip = VideoFileClip(video_path)
    duration = clip.duration
    
    for i in range(len(selected_segments)):
        start, end = selected_segments[i]
        # Add buffer before and after
        start = max(0, start - buffer_time)
        end = min(duration, end + buffer_time)
        selected_segments[i] = [start, end]
    
    # Ensure we don't exceed max_duration
    total_duration = sum(end - start for start, end in selected_segments)
    if total_duration > max_duration:
        print(f"Tổng thời lượng ({total_duration:.1f}s) vượt quá giới hạn ({max_duration}s), cắt bớt...")
        # Keep adding segments until we reach max_duration
        final_segments = []
        current_duration = 0
        
        for start, end in selected_segments:
            segment_duration = end - start
            if current_duration + segment_duration <= max_duration:
                final_segments.append([start, end])
                current_duration += segment_duration
            else:
                # Add partial segment if possible
                remaining = max_duration - current_duration
                if remaining >= 3:  # Only add if at least 3 seconds remain
                    final_segments.append([start, start + remaining])
                break
        selected_segments = final_segments
        
    # Extract and concatenate segments
    if selected_segments:
        clips = []
        for i, (start, end) in enumerate(selected_segments):
            print(f"Cảnh {i+1}: {start:.1f}s - {end:.1f}s (độ dài: {end-start:.1f}s)")
            clips.append(clip.subclip(start, end))
        
        if clips:
            print(f"Đang tạo video highlight với {len(clips)} cảnh...")
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(output_path, codec="libx264", threads=4)
            clip.close()
            final_clip.close()
            print(f"✅ Đã tạo video highlight: {output_path}")
            return True
    
    print("❌ Không thể tạo video highlight")
    clip.close()
    return False

def detect_motion_with_optical_flow(video_path, sample_rate=2, threshold=0.5, min_segment_duration=2.0):
    """
    Phát hiện chuyển động bằng Farneback Optical Flow, phân biệt chuyển động nhân vật và máy quay.
    
    Args:
        video_path: Đường dẫn video cần phân tích
        sample_rate: Số frame lấy mẫu mỗi giây
        threshold: Ngưỡng phát hiện chuyển động
        min_segment_duration: Thời lượng tối thiểu của đoạn chuyển động (giây)
    
    Returns:
        Danh sách các đoạn thời gian có chuyển động mạnh của nhân vật chính
    """
    print("🔍 Phát hiện chuyển động nâng cao với Optical Flow...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở file video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Tính toán frame interval để đạt được sample_rate mong muốn
    frame_interval = max(1, int(fps / sample_rate))
    print(f"Đang phân tích video: {duration:.1f} giây, {fps:.1f} fps")
    print(f"Lấy mẫu {sample_rate} frame/giây (mỗi {frame_interval} frame)")
        
    prev_frame = None
    motion_scores = []       # Điểm chuyển động tổng thể
    character_scores = []    # Điểm chuyển động của nhân vật (sau khi loại bỏ chuyển động máy quay)
    timestamps = []          # Thời điểm của mỗi frame được phân tích
    
    for frame_idx in tqdm(range(0, total_frames, frame_interval), desc="Phân tích optical flow"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Giảm nhiễu
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if prev_frame is None:
            prev_frame = gray
            timestamps.append(frame_idx / fps)
            motion_scores.append(0)
            character_scores.append(0)
            continue
        
        # Calculate Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, gray, None, 
            pyr_scale=0.5,   # Pyramid scale
            levels=3,        # Number of pyramid levels
            winsize=15,      # Window size
            iterations=3,    # Iterations
            poly_n=5,        # Polynomial size
            poly_sigma=1.2,  # Polynomial standard deviation
            flags=0
        )
        
        # Split flow into horizontal (x) and vertical (y) components
        flow_x, flow_y = flow[:, :, 0], flow[:, :, 1]
        # Calculate magnitude and angle of optical flow
        magnitude, angle = cv2.cartToPolar(flow_x, flow_y)
        
        # 1. Phân tích chuyển động toàn cục (máy quay)
        # Lấy trung bình của các vector chuyển động để ước lượng chuyển động máy quay
        mean_flow_x = np.mean(flow_x)
        mean_flow_y = np.mean(flow_y)
        
        # 2. Loại bỏ chuyển động máy quay từ optical flow
        # Tạo ma trận chuyển động toàn cục có cùng kích thước với frame
        h, w = flow_x.shape
        camera_motion_x = np.ones((h, w)) * mean_flow_x
        camera_motion_y = np.ones((h, w)) * mean_flow_y
        
        # Trừ chuyển động máy quay
        character_flow_x = flow_x - camera_motion_x
        character_flow_y = flow_y - camera_motion_y
        
        # Tính magnitude của chuyển động sau khi loại bỏ chuyển động máy quay
        character_magnitude, _ = cv2.cartToPolar(character_flow_x, character_flow_y)
        
        # 3. Tính điểm chuyển động
        # Chuyển động tổng thể
        total_motion = np.mean(magnitude)
        motion_scores.append(total_motion)
        
        # Chuyển động của nhân vật (sau khi loại bỏ chuyển động máy quay)
        character_motion = np.mean(character_magnitude)
        character_scores.append(character_motion)
        
        # Lưu timestamp
        timestamps.append(frame_idx / fps)
        
        # Cập nhật frame trước đó
        prev_frame = gray
        
    cap.release()
    
    # 4. Xác định các đoạn có chuyển động nhân vật mạnh
    # Tính ngưỡng thích nghi
    if not character_scores:
        return []
    
    # Làm mịn điểm chuyển động bằng moving average
    window_size = min(15, len(character_scores) // 5 + 1)  # Window size tối đa 15 hoặc 1/5 số lượng frame
    smoothed_scores = np.convolve(character_scores, np.ones(window_size)/window_size, mode='same')
    
    # Tính ngưỡng thích nghi dựa trên phân phối điểm
    mean_score = np.mean(smoothed_scores)
    std_score = np.std(smoothed_scores)
    adaptive_threshold = mean_score + std_score * threshold
    print(f"Phân tích điểm chuyển động: mean={mean_score:.4f}, std={std_score:.4f}")
    print(f"Ngưỡng phát hiện chuyển động: {adaptive_threshold:.4f}")
    
    # 5. Nhóm các frame liên tiếp có chuyển động mạnh thành các đoạn
    segments = []
    in_motion = False
    start_time = 0
    
    for i, score in enumerate(smoothed_scores):
        if score > adaptive_threshold and not in_motion:
            # Bắt đầu đoạn mới
            in_motion = True
            start_time = timestamps[i]
        elif (score <= adaptive_threshold * 0.7 or i == len(smoothed_scores) - 1) and in_motion:
            # Kết thúc đoạn hiện tại
            in_motion = False
            end_time = timestamps[i]
            # Chỉ lưu các đoạn đủ dài
            if end_time - start_time >= min_segment_duration:
                segments.append([start_time, end_time])
    
    # 6. Ghép các đoạn gần nhau
    if not segments:
        return []
        
    merged_segments = [segments[0]]
    merge_threshold = 2.0  # seconds
    
    for segment in segments[1:]:
        prev_segment = merged_segments[-1]
        # Nếu đoạn mới bắt đầu gần với kết thúc của đoạn trước, ghép chúng lại
        if segment[0] - prev_segment[1] < merge_threshold:
            prev_segment[1] = segment[1]  # Mở rộng đoạn trước
        else:
            merged_segments.append(segment)
            
    print(f"✅ Tìm thấy {len(merged_segments)} đoạn có chuyển động nhân vật mạnh")
    
    # 7. Hiển thị thông tin chi tiết về các đoạn
    total_motion_duration = sum(end - start for start, end in merged_segments)
    print(f"Tổng thời lượng chuyển động: {total_motion_duration:.1f}s ({total_motion_duration/duration*100:.1f}% của video)")
    for i, (start, end) in enumerate(merged_segments):
        print(f"  Đoạn {i+1}: {start:.1f}s - {end:.1f}s (thời lượng: {end-start:.1f}s)")
        
    return merged_segments
    
def extract_character_motion_segments(video_path, output_path, buffer_time=1.0, sample_rate=2):
    """
    Trích xuất các đoạn video có chuyển động mạnh của nhân vật chính
    
    Args:
        video_path: Đường dẫn video gốc
        output_path: Đường dẫn video đầu ra
        buffer_time: Thời gian thêm vào trước và sau mỗi đoạn (giây)
        sample_rate: Số frame lấy mẫu mỗi giây
    
    Returns:
        True nếu thành công, False nếu thất bại
    """
    print("🎬 Đang phân tích và trích xuất các cảnh có chuyển động nhân vật...")
    
    # Phát hiện các đoạn có chuyển động nhân vật
    motion_segments = detect_motion_with_optical_flow(
        video_path,
        sample_rate=sample_rate,
        threshold=0.6,  # Ngưỡng phát hiện (số lần std)
        min_segment_duration=1.5  # Độ dài tối thiểu của đoạn (giây)
    )
    
    if not motion_segments:
        print("❌ Không tìm thấy đoạn chuyển động nào đủ mạnh")
        return False
    
    # Thêm buffer time
    clip = VideoFileClip(video_path)
    duration = clip.duration
    
    buffered_segments = []
    for start, end in motion_segments:
        buffered_start = max(0, start - buffer_time)
        buffered_end = min(duration, end + buffer_time)
        buffered_segments.append([buffered_start, buffered_end])
    
    # Trích xuất và ghép các đoạn
    try:
        # Sử dụng CUDA nếu có
        if torch.cuda.is_available():
            print("🚀 Sử dụng CUDA để tăng tốc xuất video")
            success = fast_extract_segments_cuda(video_path, output_path, buffered_segments)
        else:
            # Dùng moviepy
            clips = []
            for i, (start, end) in enumerate(buffered_segments):
                print(f"Trích xuất đoạn {i+1}/{len(buffered_segments)}: {start:.1f}s - {end:.1f}s")
                clips.append(clip.subclip(start, end))
            
            if clips:
                final_clip = concatenate_videoclips(clips)
                final_clip.write_videofile(output_path, codec="libx264", threads=4)
                final_clip.close()
                success = True
            else:
                success = False
        clip.close()
        return success
    except Exception as e:
        print(f"❌ Lỗi khi trích xuất video: {e}")
        clip.close()
        return False

def visualize_optical_flow(video_path, output_path, sample_rate=1):
    """
    Tạo video hiển thị optical flow để phân biệt chuyển động máy quay và nhân vật
    
    Args:
        video_path: Đường dẫn video cần phân tích
        output_path: Đường dẫn video kết quả (hiển thị optical flow)
        sample_rate: Số frame lấy mẫu mỗi giây
    
    Returns:
        True nếu thành công, False nếu thất bại
    """
    print("🔄 Đang tạo video hiển thị optical flow...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở file video: {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tính toán frame interval để đạt được sample_rate mong muốn
    frame_interval = max(1, int(fps / sample_rate))
    
    # Thiết lập video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, sample_rate, (width, height))
    
    prev_frame = None
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[..., 1] = 255  # Saturation
    
    for frame_idx in tqdm(range(0, total_frames, frame_interval), desc="Tạo video optical flow"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert to grayscale and reduce noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if prev_frame is None:
            prev_frame = gray
            continue
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Split flow into x and y components
        flow_x, flow_y = flow[:, :, 0], flow[:, :, 1]
        
        # Calculate global (camera) motion
        mean_flow_x = np.mean(flow_x)
        mean_flow_y = np.mean(flow_y)
        
        # Remove camera motion
        char_flow_x = flow_x - mean_flow_x
        char_flow_y = flow_y - mean_flow_y
        
        # Convert to polar coordinates for visualization
        mag, ang = cv2.cartToPolar(char_flow_x, char_flow_y)
        
        # Map angle to hue (0-180 degrees)
        hsv[..., 0] = ang * 180 / np.pi / 2
        # Map magnitude to value
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert HSV to BGR for visualization
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Draw vector field (subsampled)
        step = 16
        for y in range(step//2, height, step):
            for x in range(step//2, width, step):
                # Draw camera motion vectors in red
                dx_camera = mean_flow_x * 5  # Scale for visualization
                dy_camera = mean_flow_y * 5
                cv2.arrowedLine(
                    frame, (x, y), 
                    (int(x + dx_camera), int(y + dy_camera)), 
                    (0, 0, 255), 1, tipLength=0.3
                )
                
                # Draw character motion vectors in green
                dx_char = char_flow_x[y, x] * 5
                dy_char = char_flow_y[y, x] * 5
                if abs(dx_char) > 1 or abs(dy_char) > 1:  # Only draw significant motion
                    cv2.arrowedLine(
                        frame, (x, y), 
                        (int(x + dx_char), int(y + dy_char)), 
                        (0, 255, 0), 1, tipLength=0.3
                    )
        
        # Add optical flow visualization as overlay
        alpha = 0.5
        overlay = cv2.addWeighted(frame, 0.7, flow_vis, 0.3, 0)
        
        # Add text showing camera motion
        cv2.putText(
            overlay, 
            f"Camera Motion: dx={mean_flow_x:.2f}, dy={mean_flow_y:.2f}", 
            (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
        
        # Add text showing frame info
        cv2.putText(
            overlay, 
            f"Frame: {frame_idx}/{total_frames} ({frame_idx/fps:.1f}s)", 
            (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        out.write(overlay)
        prev_frame = gray
        
    cap.release()
    out.release()
    print(f"✅ Video optical flow đã được tạo: {output_path}")
    return True

def detect_high_action_scenes(video_path, model=None, device="cuda", top_k=5, min_duration=2.0, merge_threshold=3.0):
    """
    Phát hiện các cảnh có hành động mạnh nhất bằng cách kết hợp phân tích từ nhiều nguồn:
    - Optical Flow: đo lường mức độ chuyển động
    - Scene Detection: phát hiện các điểm cắt cảnh
    - YOLOv5: phát hiện sự hiện diện và hoạt động của người
    
    Args:
        video_path: Đường dẫn video cần phân tích
        model: Mô hình YOLOv5 (nếu None, sẽ tải mô hình mặc định)
        device: Thiết bị xử lý ("cuda" hoặc "cpu")
        top_k: Số lượng cảnh hành động cao nhất muốn trả về
        min_duration: Thời lượng tối thiểu của mỗi đoạn hành động (giây)
        merge_threshold: Ngưỡng thời gian để ghép các cảnh gần nhau (giây)
        
    Returns:
        Danh sách các cặp (start_time, end_time) là thời điểm bắt đầu và kết thúc của cảnh hành động cao
    """
    print("🔍 Phát hiện cảnh hành động mạnh bằng phương pháp kết hợp...")
    
    # 1. Khởi tạo model YOLOv5 nếu không có
    if model is None:
        try:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.to(device)
            model.classes = [0]  # Chỉ quan tâm đến lớp người
            print("✅ Đã khởi tạo model YOLOv5 để phát hiện người")
        except Exception as e:
            print(f"⚠️ Không thể tải model YOLOv5: {e}")
            model = None
    
    # 2. Phân tích chuyển động với optical flow
    print("Bước 1/3: Phân tích chuyển động bằng Optical Flow...")
    motion_segments = detect_motion_with_optical_flow(
        video_path, 
        sample_rate=2, 
        threshold=0.6,
        min_segment_duration=min_duration
    )
    print(f"→ Đã phát hiện {len(motion_segments)} đoạn chuyển động mạnh")
    
    # 3. Phát hiện cắt cảnh 
    print("Bước 2/3: Phát hiện các điểm cắt cảnh...")
    scene_changes = detect_scene_changes(video_path, threshold=25)
    print(f"→ Đã phát hiện {len(scene_changes)} điểm cắt cảnh")
    
    # 4. Phát hiện người và hoạt động của họ với YOLOv5
    print("Bước 3/3: Phân tích hoạt động của người với YOLOv5...")
    if model is not None:
        person_segments = detect_action_with_yolo_batched(
            video_path, 
            model=model, 
            device=device, 
            batch_size=8, 
            frame_rate=1
        )
        print(f"→ Đã phát hiện {len(person_segments)} đoạn có hoạt động của người")
    else:
        person_segments = []
        print("⚠️ Không có model YOLOv5, bỏ qua phân tích hoạt động của người")
    
    # 5. Kết hợp kết quả phân tích để tạo điểm số hành động cho mỗi giây trong video
    print("Đang kết hợp phân tích từ các nguồn khác nhau...")
    
    # Lấy thông tin thời lượng video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps) + 1
    cap.release()
    
    # Khởi tạo mảng điểm hành động theo thời gian
    action_scores = np.zeros(duration)
    
    # Thêm điểm từ phân tích optical flow
    for start, end in motion_segments:
        start_idx = int(start)
        end_idx = min(int(end), duration - 1)
        action_scores[start_idx:end_idx+1] += 2.0  # Trọng số cao cho chuyển động
    
    # Thêm điểm từ phân tích cắt cảnh (tạo window quanh điểm cắt cảnh)
    scene_window = 2  # 2 giây trước và sau điểm cắt cảnh
    for change_time in scene_changes:
        time_idx = int(change_time)
        if time_idx < duration:
            window_start = max(0, time_idx - scene_window)
            window_end = min(duration - 1, time_idx + scene_window)
            # Thêm điểm cho các cảnh có cắt cảnh nhanh
            action_scores[window_start:window_end+1] += 1.0
    
    # Thêm điểm từ phân tích người
    for start, end in person_segments:
        start_idx = int(start)
        end_idx = min(int(end), duration - 1)
        action_scores[start_idx:end_idx+1] += 1.5  # Trọng số cho sự hiện diện của người
    
    # 6. Tính toán density của scene changes để phát hiện cảnh hành động (thường có cắt cảnh nhanh)
    scene_density = analyze_scene_change_density(scene_changes, window_size=5)
    for i, density in enumerate(scene_density):
        if i < duration:
            action_scores[i] += density * 1.5
    
    # 7. Làm mịn điểm số bằng moving average
    window_size = 5
    smoothed_scores = np.convolve(action_scores, np.ones(window_size)/window_size, mode='same')
    
    # 8. Phát hiện các đoạn có điểm hành động cao
    threshold = np.mean(smoothed_scores) + np.std(smoothed_scores)
    high_action_segments = []
    in_high_action = False
    start_time = 0
    
    for i, score in enumerate(smoothed_scores):
        if score > threshold and not in_high_action:
            in_high_action = True
            start_time = i
        elif (score <= threshold * 0.7 or i == len(smoothed_scores) - 1) and in_high_action:
            in_high_action = False
            end_time = i
            # Chỉ lấy các đoạn đủ dài
            if end_time - start_time >= min_duration:
                high_action_segments.append([start_time, end_time])
    
    # 9. Ghép các đoạn gần nhau
    merged_segments = []
    if high_action_segments:
        merged_segments = [high_action_segments[0]]
        
        for current in high_action_segments[1:]:
            previous = merged_segments[-1]
            if current[0] - previous[1] < merge_threshold:
                previous[1] = current[1]  # Mở rộng đoạn trước
            else:
                merged_segments.append(current)
    
    # 10. Tính điểm cho mỗi đoạn để sắp xếp theo mức độ hành động
    segment_scores = []
    for start, end in merged_segments:
        segment_duration = end - start
        # Tính điểm trung bình trong đoạn
        segment_score = np.mean(smoothed_scores[start:end+1])
        # Đoạn dài hơn được ưu tiên nếu điểm gần bằng nhau
        final_score = segment_score * (1 + min(segment_duration / 60.0, 0.5))
        segment_scores.append((final_score, [start, end]))
    
    # Sắp xếp các đoạn theo điểm từ cao xuống thấp
    segment_scores.sort(reverse=True)
    
    # Chọn top K đoạn có điểm cao nhất
    top_segments = []
    for _, segment in segment_scores[:top_k]:
        top_segments.append(segment)
    
    # Hiển thị thông tin đoạn được chọn
    print(f"🎯 Đã phát hiện {len(top_segments)} cảnh hành động mạnh nhất:")
    for i, (start, end) in enumerate(top_segments):
        duration = end - start
        print(f"  Cảnh {i+1}: {start}s - {end}s (thời lượng: {duration:.1f}s)")
    
    return top_segments

def extract_high_action_scenes(video_path, output_path, model=None, device="cuda", top_k=5, buffer_time=1.0):
    """
    Trích xuất các cảnh hành động mạnh nhất từ video thành một video mới
    
    Args:
        video_path: Đường dẫn video gốc
        output_path: Đường dẫn lưu video kết quả
        model: Mô hình YOLOv5 (nếu None, sẽ tải mô hình mặc định)
        device: Thiết bị xử lý ("cuda" hoặc "cpu")
        top_k: Số lượng cảnh hành động cao nhất muốn trích xuất
        buffer_time: Thời gian thêm vào trước và sau mỗi đoạn (giây)
    
    Returns:
        True nếu thành công, False nếu thất bại
    """
    print("🎬 Đang trích xuất các cảnh hành động mạnh nhất...")
    
    # Phát hiện các cảnh hành động mạnh nhất
    action_segments = detect_high_action_scenes(
        video_path, 
        model=model, 
        device=device, 
        top_k=top_k
    )
    
    if not action_segments:
        print("❌ Không tìm thấy cảnh hành động nào")
        return False
    
    # Thêm buffer time
    clip = VideoFileClip(video_path)
    duration = clip.duration
    
    buffered_segments = []
    for start, end in action_segments:
        buffered_start = max(0, start - buffer_time)
        buffered_end = min(duration, end + buffer_time)
        buffered_segments.append([buffered_start, buffered_end])
    
    # Trích xuất các đoạn video
    try:
        # Sử dụng GPU nếu có
        if device == "cuda" and torch.cuda.is_available():
            success = fast_extract_segments_cuda(video_path, output_path, buffered_segments)
        else:
            # Sử dụng CPU
            success = extract_segments_to_video(video_path, output_path, buffered_segments)
        
        if success:
            print(f"✅ Đã trích xuất thành công {len(buffered_segments)} cảnh hành động vào: {output_path}")
        else:
            print("❌ Không thể trích xuất các cảnh hành động")
        return success
    except Exception as e:
        print(f"❌ Lỗi khi trích xuất cảnh hành động: {e}")
        return False

def extract_highlight_clips(video_path, timestamps, output_path, buffer_time=1.0, use_cuda=True):
    """
    Trích xuất các đoạn highlight từ video dựa trên timestamps và ghép thành video mới
    
    Args:
        video_path: Đường dẫn đến video gốc
        timestamps: Danh sách các cặp [start_time, end_time]
        output_path: Đường dẫn lưu video kết quả
        buffer_time: Thời gian bổ sung trước và sau mỗi đoạn (giây)
        use_cuda: Sử dụng GPU để tăng tốc nếu có
    
    Returns:
        True nếu thành công, False nếu thất bại
    """
    if not timestamps:
        print("❌ Không có timestamps để trích xuất")
        return False
    
    print(f"📋 Chuẩn bị trích xuất {len(timestamps)} đoạn highlights...")
    
    # Kiểm tra thông tin video để đảm bảo timestamps hợp lệ
    clip = VideoFileClip(video_path)
    duration = clip.duration
    
    # Thêm buffer time và đảm bảo timestamps nằm trong giới hạn của video
    buffered_segments = []
    total_extract_duration = 0
    
    for start, end in timestamps:
        buffered_start = max(0, start - buffer_time)
        buffered_end = min(duration, end + buffer_time)
        # Chỉ thêm các đoạn có độ dài hợp lý
        if buffered_end - buffered_start >= 1.0:
            buffered_segments.append([buffered_start, buffered_end])
            total_extract_duration += (buffered_end - buffered_start)
    
    if not buffered_segments:
        print("❌ Không có đoạn nào hợp lệ sau khi thêm buffer time")
        clip.close()
        return False
    
    # Sắp xếp các đoạn theo thứ tự thời gian
    buffered_segments.sort(key=lambda x: x[0])
    
    # Hiển thị thông tin trích xuất
    print(f"🎬 Trích xuất {len(buffered_segments)} đoạn highlight (tổng thời lượng: {total_extract_duration:.1f}s)")
    
    # Thử sử dụng GPU để trích xuất nếu có thể
    if use_cuda and torch.cuda.is_available():
        print("🚀 Sử dụng GPU để tăng tốc trích xuất video")
        success = fast_extract_segments_cuda(video_path, output_path, buffered_segments)
        if success:
            print(f"✅ Đã tạo video highlight thành công: {output_path}")
            clip.close()
            return True
        else:
            print("⚠️ Trích xuất với GPU thất bại, chuyển sang phương pháp thông thường")
        
    # Sử dụng phương pháp thông thường (CPU)
    try:
        clips = []
        for i, (start, end) in enumerate(buffered_segments):
            print(f"  Đoạn {i+1}/{len(buffered_segments)}: {start:.1f}s - {end:.1f}s (độ dài: {end-start:.1f}s)")
            clips.append(clip.subclip(start, end))
        
        if clips:
            print(f"🔄 Đang ghép {len(clips)} đoạn video...")
            final_clip = concatenate_videoclips(clips)
            # Xuất video với codec H.264
            print(f"💾 Đang xuất video highlight ra {output_path}...")
            final_clip.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac",
                threads=min(4, os.cpu_count() or 2),
                preset="medium"  # Cân bằng giữa tốc độ xuất và chất lượng
            )
            final_clip.close()
            clip.close()
            print(f"✅ Đã tạo video highlight thành công: {output_path}")
            return True
        else:
            clip.close()
            print("❌ Không có đoạn video nào hợp lệ để ghép")
            return False
    except Exception as e:
        print(f"❌ Lỗi khi tạo video highlight: {e}")
        try:
            clip.close()
        except:
            pass
        return False

def initialize_yolo8_model(device="cuda", model_name="yolov8s.pt", task="detect"):
    """
    Initialize YOLOv8 model for object detection or action recognition
    
    Args:
        device: Device to run the model on ("cuda" or "cpu")
        model_name: Name or path of the model to load ("yolov8s.pt", "yolov8n-pose.pt", etc.)
        task: Type of task ("detect", "pose", "seg", "classify")
        
    Returns:
        Initialized YOLOv8 model
    """
    print(f"🔍 Initializing YOLOv8 model for {task}...")
    try:
        from ultralytics import YOLO
        
        # Check if model exists or download it
        model_path = os.path.join('models', model_name)
        if not os.path.exists(model_path):
            model_path = model_name  # Use direct name for downloading
        
        # Load YOLOv8 model
        model = YOLO(model_path)
        
        # Set device
        if device == "cuda" and torch.cuda.is_available():
            model.to(device)
            print(f"✅ YOLOv8 model loaded on {device}")
        else:
            print("⚠️ CUDA not available, using CPU")
        return model
    except Exception as e:
        print(f"❌ Failed to load YOLOv8: {e}")
        return None

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
            return detect_high_action_scenes(video_path, None, device, top_k, min_duration, merge_threshold)
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    
    # Extract frames with Decord for faster processing
    sample_rate = 1  # 1 frame per second
    frames, timestamps, _ = None, None, None  # Initialize with None values
    if DECORD_AVAILABLE:
        try:
            # FIX: Handle 3 return values properly
            frames, timestamps, fps = extract_frames_with_decord(
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
    
    # Process frames in batches with YOLOv8
    batch_size = 8
    yolo_results = []
    
    print("🔍 Analyzing frames with YOLOv8...")
    for i in tqdm(range(0, len(frames), batch_size)):
        batch_frames = frames[i:i+batch_size]
        
        # Process batch with YOLOv8
        batch_results = model.predict(
            batch_frames, 
            conf=confidence, 
            verbose=False
        )
        
        # Store results
        for j, result in enumerate(batch_results):
            idx = i + j
            if idx < len(timestamps):
                # For pose detection, we get keypoints
                if task == "pose":
                    keypoints = result.keypoints
                    num_people = len(result.boxes) if hasattr(result, 'boxes') else 0
                    yolo_results.append({
                        'timestamp': timestamps[idx],
                        'num_people': num_people,
                        'has_keypoints': keypoints is not None and len(keypoints) > 0
                    })
                # For object detection
                else:
                    # Count people (class 0)
                    boxes = result.boxes
                    people = [box for box in boxes if int(box.cls) == 0]
                    yolo_results.append({
                        'timestamp': timestamps[idx],
                        'num_people': len(people),
                        'boxes': boxes
                    })
    
    # Start parallel optical flow analysis
    print("🔄 Starting optical flow analysis...")
    motion_segments = detect_motion_with_optical_flow(
        video_path, 
        sample_rate=2,  # Higher sample rate for optical flow
        threshold=0.6,
        min_segment_duration=min_duration
    )
    
    # Calculate action scores from YOLOv8 results
    action_scores = []
    for result in yolo_results:
        # More people = higher score
        people_score = min(result['num_people'] * 0.5, 3.0)  # Cap at 3.0
        
        # Add results to list with timestamp
        action_scores.append({
            'timestamp': result['timestamp'],
            'score': people_score,
        })
    
    # Combine action scores with optical flow results to create final scores
    # Create a time-indexed score array
    time_range = np.arange(0, int(duration) + 1)
    combined_scores = np.zeros(len(time_range))
    
    # Add YOLOv8 scores
    for result in action_scores:
        time_idx = int(result['timestamp'])
        if time_idx < len(combined_scores):
            # Add people score
            combined_scores[time_idx] += result['score']
        
    # Add optical flow scores
    for start, end in motion_segments:
        start_idx = int(start)
        end_idx = min(int(end), len(combined_scores) - 1)
        combined_scores[start_idx:end_idx+1] += 2.0  # Higher weight for motion
    
    # Smooth scores
    window_size = 5
    smoothed_scores = np.convolve(combined_scores, np.ones(window_size)/window_size, mode='same')
    
    # Find high action segments
    threshold = np.mean(smoothed_scores) + np.std(smoothed_scores)
    high_action_segments = []
    in_high_action = False
    start_time = 0
    
    for i, score in enumerate(smoothed_scores):
        if score > threshold and not in_high_action:
            in_high_action = True
            start_time = i
        elif (score <= threshold * 0.7 or i == len(smoothed_scores) - 1) and in_high_action:
            in_high_action = False
            end_time = i
            if end_time - start_time >= min_duration:
                high_action_segments.append([start_time, end_time])
    
    # Merge segments that are close to each other
    merged_segments = []
    if high_action_segments:
        merged_segments = [high_action_segments[0]]
        
        for current in high_action_segments[1:]:
            previous = merged_segments[-1]
            if current[0] - previous[1] < merge_threshold:
                previous[1] = current[1]
            else:
                merged_segments.append(current)
    
    # Score segments for ranking
    segment_scores = []
    for start, end in merged_segments:
        segment_duration = end - start
        # Average score in the segment
        segment_score = np.mean(smoothed_scores[start:end+1])
        # Slightly favor longer segments with similar scores
        final_score = segment_score * (1 + min(segment_duration / 60.0, 0.5))
        segment_scores.append((final_score, [start, end]))
    
    # Sort by score and get top segments
    segment_scores.sort(reverse=True)
    top_segments = []
    for _, segment in segment_scores[:top_k]:
        top_segments.append(segment)
    
    # Show selected segments
    print(f"🎯 Found {len(top_segments)} top action scenes:")
    for i, (start, end) in enumerate(top_segments):
        duration = end - start
        print(f"  Scene {i+1}: {start}s - {end}s (duration: {duration:.1f}s)")
    
    return top_segments

def create_single_action_short(video_path, output_path, model=None, device="cuda", 
                             buffer_time=1.0, min_duration=5.0, max_duration=15.0):
    """Create a short video with a single high-action moment"""
    print("🎬 Creating single action highlight video...")
    
    # Check GPU availability again to make sure
    if device == "cuda" and not torch.cuda.is_available():
        force_cuda = os.environ.get('FORCE_CUDA', '0') == '1'
        if not force_cuda:
            print("⚠️ GPU không khả dụng, chuyển sang CPU")
            device = "cpu"
    
    # Initialize YOLOv8 model if not provided
    if model is None:
        from yolo8_detection import initialize_yolo8_model
        model = initialize_yolo8_model(device)
        if model is None:
            print("❌ Không thể khởi tạo YOLOv8 model")
            return False
    
    # Rest of the function...

def create_multi_action_short(video_path, output_path, model=None, device="cuda", 
                            target_duration=30.0, max_scenes=5, buffer_time=1.0,
                            min_segment_duration=3.0):
    """
    Create a short video with multiple high-action moments merged together
    
    Args:
        video_path: Path to the input video
        output_path: Path to save the output video
        model: YOLOv8 model (if None, will initialize one)
        device: Device to run the model on
        target_duration: Target duration of the final video
        max_scenes: Maximum number of scenes to include
        buffer_time: Time to add before and after each segment
        min_segment_duration: Minimum duration for each action segment
        
    Returns:
        True if successful, False otherwise
    """
    print(f"🎬 Creating multi-action short video (target duration: {target_duration:.1f}s)...")
    
    # Initialize YOLOv8 if not provided
    if model is None:
        model = initialize_yolo8_model(device)
    
    # Get top action segments
    segments = detect_high_action_scenes_yolo8(
        video_path, model, device, top_k=max_scenes, 
        min_duration=min_segment_duration
    )
    
    if not segments:
        print("❌ No action scenes detected")
        return False
    
    # Get video duration
    cap = cv2.VideoCapture(video_path)
    video_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    # Add buffer time to segments
    buffered_segments = []
    for start, end in segments:
        buffered_start = max(0, start - buffer_time)
        buffered_end = min(video_duration, end + buffer_time)
        buffered_segments.append([buffered_start, buffered_end])
    
    # Calculate total duration and trim if necessary
    total_duration = sum(end - start for start, end in buffered_segments)
    if total_duration > target_duration:
        print(f"⚠️ Total duration ({total_duration:.1f}s) exceeds target ({target_duration:.1f}s), trimming...")
        # Prioritize keeping the highest-scored segments
        selected_segments = []
        current_duration = 0
        
        for start, end in buffered_segments:
            segment_duration = end - start
            if current_duration + segment_duration <= target_duration:
                selected_segments.append([start, end])
                current_duration += segment_duration
            else:
                # If we can fit part of this segment, trim it
                remaining_time = target_duration - current_duration
                if remaining_time >= min_segment_duration:
                    # Trim from the end to preserve the action intro
                    selected_segments.append([start, start + remaining_time])
                    current_duration += remaining_time
                break
        
        buffered_segments = selected_segments
    
    # Extract and combine segments
    print(f"📹 Extracting {len(buffered_segments)} segments (total: {total_duration:.1f}s)...")
    
    # Use GPU acceleration if available
    if device == "cuda" and torch.cuda.is_available():
        success = fast_extract_segments_cuda(video_path, output_path, buffered_segments)
    else:
        # Use CPU with moviepy
        try:
            clip = VideoFileClip(video_path)
            clips = []
            
            for i, (start, end) in enumerate(buffered_segments):
                print(f"  Segment {i+1}: {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s)")
                clips.append(clip.subclip(start, end))
            
            if clips:
                final_clip = concatenate_videoclips(clips)
                final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
                final_clip.close()
                clip.close()
                success = True
            else:
                success = False
        except Exception as e:
            print(f"❌ Error creating video: {e}")
            success = False
    
    if success:
        print(f"✅ Successfully created multi-action short: {output_path}")
        return True
    else:
        print("❌ Failed to create short video")
        return False

def extract_highlight_clips(video_path, timestamps, output_path, buffer_time=1.0, use_cuda=True):
    """
    Trích xuất các đoạn highlight từ video dựa trên timestamps và ghép thành video mới
    
    Args:
        video_path: Đường dẫn đến video gốc
        timestamps: Danh sách các cặp [start_time, end_time]
        output_path: Đường dẫn lưu video kết quả
        buffer_time: Thời gian bổ sung trước và sau mỗi đoạn (giây)
        use_cuda: Sử dụng GPU để tăng tốc nếu có
    
    Returns:
        True nếu thành công, False nếu thất bại
    """
    # ...existing code...

def detect_motion_with_optical_flow_cpp(video_path, sample_rate=2, threshold=0.5):
    """
    Enhanced version of detect_motion_with_optical_flow using C++ acceleration
    """
    print("🔍 Analyzing video motion using C++ accelerated optical flow...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval based on sample rate
    frame_interval = max(1, int(fps / sample_rate))
    
    prev_frame = None
    motion_scores = []
    timestamps = []
    
    for frame_idx in tqdm(range(0, total_frames, frame_interval), desc="Analyzing motion"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None:
            prev_frame = gray
            timestamps.append(frame_idx / fps)
            motion_scores.append(0)
            continue
        
        # Calculate optical flow using optimized function
        flow_x, flow_y = calculate_optical_flow_optimized(prev_frame, gray)
        
        # Calculate magnitude and global motion
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        mean_flow_x = np.mean(flow_x)
        mean_flow_y = np.mean(flow_y)
        
        # Compute character motion by removing camera motion
        character_flow_x = flow_x - mean_flow_x
        character_flow_y = flow_y - mean_flow_y
        character_magnitude = np.sqrt(character_flow_x**2 + character_flow_y**2)
        character_motion = np.mean(character_magnitude)
        
        motion_scores.append(character_motion)
        timestamps.append(frame_idx / fps)
        prev_frame = gray
        
    cap.release()
    
    # Rest of the function (finding segments, etc.) remains the same

def calculate_optical_flow_optimized(prev_frame, curr_frame, device="cuda"):
    """
    Calculate optical flow using C++ extension if available, otherwise fall back to OpenCV
    """
    try:
        from utils.gpu_utils import try_load_cpp_extensions
        cpp_extensions = try_load_cpp_extensions()
        
        if "optical_flow" in cpp_extensions:
            optical_flow_cpp = cpp_extensions["optical_flow"]
            use_cuda = device == "cuda" and torch.cuda.is_available()
            device_id = 0 if use_cuda else -1
            print("Using C++ optical flow implementation")
            flow_x, flow_y = optical_flow_cpp.calculate_flow(
                prev_frame, curr_frame, use_cuda, device_id)
            return flow_x, flow_y
    except Exception as e:
        print(f"Error using C++ optical flow: {e}")
        print("Falling back to OpenCV optical flow implementation")
    
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow[:, :, 0], flow[:, :, 1]
