import os
import torch
import whisper
import moviepy.editor as mp
from tqdm import tqdm
import concurrent.futures
import numpy as np
import re  # Thêm thư viện re cho clean_transcript
import subprocess
import shutil
import signal

# Import các hàm cần thiết từ các module khác
# Nếu cần import circular, dùng import điều kiện
def import_from_modules():
    global fast_motion_detection, fast_motion_detection_with_decord, fast_extract_segments, fast_extract_segments_cuda
    try:
        from action_detection import (
            fast_motion_detection, 
            fast_motion_detection_with_decord, 
            fast_extract_segments, 
            fast_extract_segments_cuda
        )
    except ImportError:
        print("⚠️ Không thể import các hàm từ action_detection")
    
    global filter_irrelevant_content, enhanced_summarize_transcript
    try:
        from summary_generator import filter_irrelevant_content, enhanced_summarize_transcript
    except ImportError:
        print("⚠️ Không thể import các hàm từ summary_generator")

def extract_audio(video_path, audio_path):
    """Trích xuất âm thanh từ video"""
    print("Đang trích xuất âm thanh...")
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    print("✅ Trích xuất âm thanh hoàn tất")
    return audio_path

def filter_audio(input_audio_path, output_audio_path):
    """Lọc nhiễu âm thanh để cải thiện chất lượng nhận dạng"""
    import subprocess
    import shutil
    
    print("Đang lọc nhiễu âm thanh...")
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        cmd = ["ffmpeg", "-i", input_audio_path, "-af", "afftdn=nf=-25", "-y", output_audio_path]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ Lọc nhiễu âm thanh thành công")
        return True
    except subprocess.CalledProcessError:
        print("⚠️ Không thể lọc âm thanh, ffmpeg không khả dụng hoặc gặp lỗi")
        shutil.copy(input_audio_path, output_audio_path)
        return False
    except Exception as e:
        print(f"❌ Lỗi khi lọc âm thanh: {e}")
        shutil.copy(input_audio_path, output_audio_path)
        return False

def initialize_whisper_model(device, model_size="base"):
    """Khởi tạo model Whisper phù hợp với phần cứng"""
    print(f"Đang khởi tạo model Whisper trên {device}...")
    
    if torch.cuda.is_available():
        # Kiểm tra bộ nhớ GPU để chọn kích thước model phù hợp
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        
        if gpu_memory >= 8.0:
            whisper_model_size = "medium"
            print(f"🔍 GPU có {gpu_memory:.1f}GB RAM, sử dụng model Whisper {whisper_model_size}")
        else:
            whisper_model_size = "base"
            print(f"⚠️ GPU chỉ có {gpu_memory:.1f}GB RAM, sử dụng model Whisper {whisper_model_size}")
            
        try:
            # Sử dụng FP16 để tăng tốc và tiết kiệm bộ nhớ
            whisper_model = whisper.load_model(whisper_model_size, device=device)
            print("✅ Đã khởi tạo Whisper trên GPU")
        except Exception as e:
            print(f"❌ Lỗi khi tải model Whisper: {e}")
            print("Thử tải model nhỏ hơn...")
            whisper_model = whisper.load_model("base", device=device)
    else:
        whisper_model = whisper.load_model("base", device="cpu")
        print("✅ Đã khởi tạo Whisper trên CPU")
        
    return whisper_model

def transcribe_audio(audio_path, whisper_model):
    """Chuyển đổi giọng nói thành văn bản bằng Whisper AI"""
    print(f"Đang chuyển đổi âm thanh thành văn bản (có thể mất vài phút)...")
    
    # Giải phóng bộ nhớ GPU trước khi chạy Whisper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Sử dụng FP16 để tối ưu hóa
    result = whisper_model.transcribe(audio_path, fp16=True)
    
    return result["text"]

def transcribe_audio_with_timestamps(audio_path, whisper_model):
    """Chuyển đổi âm thanh thành văn bản với timestamp cho mỗi câu"""
    print(f"Đang chuyển đổi âm thanh thành văn bản với timestamps...")
    
    # Giải phóng bộ nhớ GPU trước khi chạy Whisper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    try:
        # Sử dụng tham số word_timestamps=True để lấy timestamp cho mỗi từ
        result = whisper_model.transcribe(
            audio_path, 
            fp16=True,
            word_timestamps=True,
            verbose=False
        )
        
        # Trả về toàn bộ kết quả bao gồm cả segments
        return result
        
    except Exception as e:
        print(f"Lỗi khi transcribe với timestamps: {e}")
        # Phương pháp dự phòng - không có timestamps
        result = whisper_model.transcribe(audio_path, fp16=True)
        return result

def transcribe_audio_optimized(audio_path, whisper_model=None, model_size="medium", chunk_length=30, has_pydub=True):
    """Phương pháp tối ưu để chuyển đổi âm thanh thành văn bản"""
    print("Bắt đầu phương pháp chuyển đổi âm thanh tối ưu...")
    
    # Luôn thử dùng Faster-Whisper trước nếu có GPU
    if torch.cuda.is_available():
        try:
            print("🚀 Sử dụng Faster-Whisper (GPU) để tăng tốc 3-5 lần...")
            result = transcribe_audio_faster(audio_path, language="auto")
            if result and result["text"]:
                return result["text"]
            print("Faster-Whisper không thành công, chuyển sang phương pháp song song...")
        except Exception as e:
            print(f"Lỗi khi dùng Faster-Whisper: {e}")
    
    # Thử dùng phương pháp song song thứ hai
    try:
        result = transcribe_audio_parallel(
            audio_path, 
            language="auto", 
            chunk_length_sec=15,
            max_workers=4
        )
        if result and result["text"]:
            return result["text"]
        print("Phương pháp song song không thành công, chuyển sang phương pháp cũ...")
    except Exception as e:
        print(f"Lỗi khi dùng phương pháp song song: {e}")
        print("Chuyển sang phương pháp cũ...")
    
    # Nếu Faster-Whisper và phương pháp song song không thành công, dùng phương pháp cũ
    if not has_pydub:
        print("⚠️ Không tìm thấy thư viện pydub, sử dụng phương pháp thông thường...")
        if whisper_model is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            whisper_model = whisper.load_model("base", device=device)
        return transcribe_audio(audio_path, whisper_model)
        
    from pydub import AudioSegment
    
    # Tạo thư mục tạm để lưu các file âm thanh đã chia nhỏ
    temp_dir = os.path.join('output_videos', 'temp_audio_chunks')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Chia nhỏ file âm thanh
    print(f"Đang chia nhỏ file âm thanh thành các đoạn {chunk_length}s...")
    audio = AudioSegment.from_file(audio_path)
    duration_seconds = len(audio) / 1000
    chunks_count = int(duration_seconds / chunk_length) + 1
    
    # Tạo danh sách chứa đường dẫn các file tạm
    chunk_files = []
    
    # Chia nhỏ và lưu các đoạn âm thanh
    for i in range(chunks_count):
        start_ms = i * chunk_length * 1000
        end_ms = min((i + 1) * chunk_length * 1000, len(audio))
        chunk = audio[start_ms:end_ms]
        
        # Lưu đoạn tạm thời
        chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.wav")
        chunk.export(chunk_path, format="wav")
        chunk_files.append(chunk_path)
    
    # Tạo hàm để xử lý từng đoạn
    def process_chunk(chunk_file):
        # Giải phóng bộ nhớ GPU nếu cần
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Sử dụng mô hình toàn cục thay vì tạo mô hình mới cho từng đoạn
        if whisper_model is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            local_model = whisper.load_model("base", device=device)
            result = local_model.transcribe(chunk_file, fp16=torch.cuda.is_available())
        else:
            result = whisper_model.transcribe(chunk_file, fp16=torch.cuda.is_available())
        
        # Giải phóng bộ nhớ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return result["text"]
    
    # Xử lý tuần tự tất cả các đoạn audio
    print(f"Xử lý {len(chunk_files)} đoạn âm thanh tuần tự...")
    results = []
    for chunk_file in tqdm(chunk_files, desc="Nhận dạng giọng nói"):
        results.append(process_chunk(chunk_file))
    
    # Gộp kết quả và xử lý văn bản
    transcript = " ".join(results)
    
    # Xóa các file tạm
    for chunk_file in chunk_files:
        os.remove(chunk_file)
        
    # Xóa thư mục tạm
    try:
        os.rmdir(temp_dir)
    except:
        pass
        
    return transcript

def generate_voice_over(text, output_path, lang='vi'):
    """Generate voice-over audio file from text using Google Text-to-Speech"""
    try:
        from gtts import gTTS
        print("Đang tạo voice-over từ văn bản...")
        tts = gTTS(text=text, lang=lang)
        tts.save(output_path)
        print(f"✅ Đã tạo voice-over thành công: {output_path}")
        return True
    except ImportError:
        print("⚠️ Không tìm thấy thư viện gtts, vui lòng cài đặt: pip install gtts")
        return False
    except Exception as e:
        print(f"❌ Lỗi khi tạo voice-over: {e}")
        return False

def initialize_faster_whisper():
    """Khởi tạo Faster Whisper model với compute type phù hợp"""
    try:
        from faster_whisper import WhisperModel
        
        print("Khởi tạo Faster Whisper model...")
        # Kiểm tra GPU
        if torch.cuda.is_available():
            # Kiểm tra bộ nhớ GPU để chọn kích thước model phù hợp
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            
            if gpu_memory >= 8.0:
                model_size = "medium"
            else:
                model_size = "small"
            
            # Dùng float32 thay vì float16 cho GTX 1060 và các GPU cũ hơn
            try:
                # Thử trước với float32
                print(f"🔍 GPU có {gpu_memory:.1f}GB RAM, sử dụng model {model_size} với float32")
                model = WhisperModel(model_size, device="cuda", compute_type="float32")
                return model
            except Exception:
                # Nếu float32 thất bại, dùng int8
                print(f"🔍 GPU có {gpu_memory:.1f}GB RAM, sử dụng model {model_size} với int8")
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                return model
        else:
            print("⚠️ Không phát hiện GPU, sử dụng CPU với model small")
            model = WhisperModel("small", device="cpu", compute_type="int8")
            return model
    except ImportError:
        print("⚠️ Không tìm thấy Faster Whisper, sử dụng Whisper thông thường")
        return None
    except Exception as e:
        print(f"⚠️ Lỗi khi khởi tạo Faster Whisper: {e}")
        return None

def transcribe_audio_faster(audio_path, language="auto"):
    """Sử dụng Faster-Whisper để nhận dạng - tăng tốc 3-5x"""
    try:
        model = initialize_faster_whisper()
        if model is None:
            # Fallback to standard whisper
            return None
            
        print(f"Đang chuyển đổi âm thanh thành văn bản với Faster Whisper...")
        
        # Fix the language parameter issue - use None for auto detection instead of "auto"
        detect_language = None if language == "auto" else language
        
        segments, info = model.transcribe(audio_path, beam_size=5, language=detect_language)
        
        # Print the detected language
        print(f"Đã phát hiện ngôn ngữ: {info.language} với độ tin cậy {info.language_probability:.2f}")
        
        # Collect all segments
        result = {"text": "", "segments": []}
        for segment in segments:
            result["text"] += segment.text + " "
            result["segments"].append({
                "text": segment.text,
                "start": segment.start,
                "end": segment.end
            })
        
        return result
    except Exception as e:
        print(f"❌ Lỗi khi sử dụng Faster Whisper: {e}")
        return None

def split_audio_into_chunks(audio_path, chunk_length_ms=15000, output_dir=None):
    """Chia file âm thanh thành nhiều đoạn nhỏ để xử lý song song"""
    from pydub import AudioSegment
    
    if output_dir is None:
        output_dir = os.path.join('output_videos', 'audio_chunks')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Đang chia audio thành các đoạn {chunk_length_ms/1000}s...")
    audio = AudioSegment.from_file(audio_path)
    
    # Tính số lượng chunk
    chunk_files = []
    for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
        end = min(start + chunk_length_ms, len(audio))
        chunk = audio[start:end]
        # Mở rộng chunk để tránh cắt giữa từ (nếu không phải chunk cuối)
        if end < len(audio):
            # Thêm 500ms vào cuối mỗi chunk để chồng lấp
            end = min(end + 500, len(audio))
            chunk = audio[start:end]
        chunk_file = os.path.join(output_dir, f"chunk_{i:03d}.wav")
        chunk.export(chunk_file, format="wav")
        chunk_files.append({
            'path': chunk_file,
            'start_ms': start,
            'end_ms': end,
            'index': i
        })
        
    return chunk_files

def process_audio_chunk(chunk_info, model, language="auto"):
    """Xử lý một chunk audio với Faster Whisper"""
    try:
        print(f"Xử lý chunk {chunk_info['index']}...")
        
        if isinstance(model, str) and model == "standard_whisper":
            # Sử dụng Whisper thường
            import whisper
            whisper_model = whisper.load_model("base")
            result = whisper_model.transcribe(chunk_info['path'], language=language)
            text = result["text"]
            return {
                'text': text,
                'start_time': chunk_info['start_ms'] / 1000,
                'index': chunk_info['index']
            }
        else:
            # Sử dụng Faster Whisper
            segments, _ = model.transcribe(
                chunk_info['path'], 
                beam_size=5,
                language=language,  # Use None instead of "auto" for Faster Whisper
                word_timestamps=True
            )
            
            # Collect results from this chunk
            text_segments = []
            for segment in segments:
                segment_dict = {
                    'text': segment.text,
                    'start': segment.start + chunk_info['start_ms']/1000,
                    'end': segment.end + chunk_info['start_ms']/1000,
                    'words': []
                }
                # Add word timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_dict = {
                            'word': word.word,
                            'start': word.start + chunk_info['start_ms']/1000,
                            'end': word.end + chunk_info['start_ms']/1000,
                            'probability': word.probability
                        }
                        segment_dict['words'].append(word_dict)
                text_segments.append(segment_dict)
            
            # Combine text from all segments
            full_text = " ".join([s['text'] for s in text_segments])
            return {
                'text': full_text,
                'segments': text_segments,
                'start_time': chunk_info['start_ms'] / 1000,
                'index': chunk_info['index']
            }
    except Exception as e:
        print(f"Lỗi khi xử lý chunk {chunk_info['index']}: {e}")
        return {
            'text': "",
            'start_time': chunk_info['start_ms'] / 1000,
            'index': chunk_info['index'],
            'error': str(e)
        }

def transcribe_audio_parallel(audio_path, language="auto", chunk_length_sec=15, max_workers=4):
    """Chuyển đổi âm thanh thành văn bản nhanh hơn với xử lý song song"""
    print("🚀 Đang chuyển đổi âm thanh thành văn bản với xử lý song song...")
    
    # Tạo model Faster Whisper hoặc sử dụng Whisper thường
    model = initialize_faster_whisper()
    if model is None:
        model = "standard_whisper"
        print("Sử dụng Whisper tiêu chuẩn để xử lý song song các chunk")
    
    # Chia nhỏ file âm thanh
    chunks = split_audio_into_chunks(audio_path, chunk_length_ms=chunk_length_sec*1000)
    print(f"Đã chia thành {len(chunks)} chunk, mỗi chunk dài {chunk_length_sec}s")
    
    # Điều chỉnh số workers để không quá tải system
    max_workers = min(max_workers, len(chunks), os.cpu_count())
    print(f"Sử dụng {max_workers} workers để xử lý song song với ngôn ngữ={language}")
    
    # Xử lý song song các chunk
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use partial to ensure we pass the correct language parameter
        import functools
        process_chunk_func = functools.partial(process_audio_chunk, model=model, language=language)
        
        future_to_chunk = {
            executor.submit(process_chunk_func, chunk): chunk
            for chunk in chunks
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_chunk), 
                          total=len(chunks), 
                          desc="Xử lý các chunk âm thanh"):
            chunk = future_to_chunk[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Chunk {chunk['index']} gặp lỗi: {e}")
    
    # Sắp xếp kết quả theo thứ tự chunk
    results.sort(key=lambda x: x['index'])
    
    # Gộp các kết quả
    full_transcript = ""
    all_segments = []
    for result in results:
        if 'error' not in result:
            full_transcript += result['text'] + " "
            if 'segments' in result:
                all_segments.extend(result['segments'])
    
    # Clean up temporary files
    for chunk in chunks:
        try:
            os.remove(chunk['path'])
        except:
            pass
    
    # Xóa thư mục tạm nếu trống
    try:
        chunk_dir = os.path.dirname(chunks[0]['path'])
        if not os.listdir(chunk_dir):
            os.rmdir(chunk_dir)
    except:
        pass
    
    # Tạo đối tượng kết quả tương thích với whisper
    final_result = {
        "text": full_transcript.strip(),
        "segments": all_segments
    }
    
    return final_result

# Thêm hàm clean_transcript
def clean_transcript(text):
    """Làm sạch transcript, loại bỏ các ký tự và từ ngữ không cần thiết"""
    if not text:
        return ""
    
    # Chuyển đổi xuống dòng thành dấu cách
    text = re.sub(r'\n+', ' ', text)
    
    # Loại bỏ dấu cách thừa
    text = re.sub(r'\s+', ' ', text)
    
    # Loại bỏ các cụm từ không cần thiết (thường từ Whisper)
    unnecessary_phrases = [
        r'\[Music\]', r'\[music\]', r'\[Âm nhạc\]', r'\[âm nhạc\]',
        r'\[Applause\]', r'\[applause\]', r'\[Tiếng vỗ tay\]',
        r'\[Background noise\]', r'\[Tiếng ồn nền\]',
        r'\[Laughter\]', r'\[laughter\]', r'\[Tiếng cười\]'
    ]
    for phrase in unnecessary_phrases:
        text = re.sub(phrase, '', text)
    
    # Loại bỏ dấu ngoặc vuông và nội dung bên trong [...]
    text = re.sub(r'\[.*?\]', '', text)
    
    # Loại bỏ lại dấu cách thừa sau khi đã xóa nội dung không cần thiết
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Thêm phương pháp dự phòng cho các hàm có thể không tồn tại
def dummy_filter_irrelevant_content(text):
    """Phương pháp dự phòng khi không import được từ summary_generator"""
    return text

def dummy_enhanced_summarize_transcript(text, ratio=0.3):
    """Phương pháp dự phòng khi không import được từ summary_generator"""
    lines = text.split('.')
    num_lines = max(1, int(len(lines) * ratio))
    return '. '.join(lines[:num_lines]) + ('.' if not lines[0].endswith('.') else '')

# Sửa hàm process_review_mode_with_timeout
def process_review_mode_with_timeout(input_path, output_path, device, model=None, ratio=0.3, timeout=300,
                                   use_faster_whisper=True, use_cuda=True):
    """Phiên bản process_review_mode với timeout để tránh xử lý quá lâu"""
    import signal
    
    # Import các hàm cần thiết từ các module khác
    import_from_modules()
    
    # Định nghĩa các hàm dự phòng nếu cần
    _filter_irrelevant_content = filter_irrelevant_content if 'filter_irrelevant_content' in globals() else dummy_filter_irrelevant_content
    _enhanced_summarize_transcript = enhanced_summarize_transcript if 'enhanced_summarize_transcript' in globals() else dummy_enhanced_summarize_transcript
    
    # Hàm xử lý khi timeout
    def timeout_handler(signum, frame):
        raise TimeoutError("Quá thời gian xử lý")
    
    # Thiết lập timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
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
                result = transcribe_audio_faster(filtered_audio_path)
                if result and result["text"]:
                    transcript = result["text"]
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
                result = transcribe_audio_parallel(
                    filtered_audio_path, 
                    chunk_length_sec=15,
                    max_workers=4
                )
                if result and result["text"]:
                    transcript = result["text"]
                    transcript = clean_transcript(transcript)
                    print("✅ Chuyển đổi âm thanh thành công với phương pháp song song")
            except Exception as e:
                print(f"⚠️ Lỗi khi dùng phương pháp song song: {e}")
                # Khởi tạo model Whisper và fallback
                print("Chuyển sang whisper thông thường...")
                whisper_model = initialize_whisper_model(device)
                transcript = transcribe_audio_optimized(filtered_audio_path, whisper_model)
                transcript = clean_transcript(transcript)
            
        # Tóm tắt transcript
        if transcript and len(transcript.split()) >= 10:
            filtered_transcript = _filter_irrelevant_content(transcript)
            summary = _enhanced_summarize_transcript(filtered_transcript, ratio)
        else:
            summary = "Không phát hiện lời thoại có nghĩa trong video."
            
        # Reset timeout cho phần 2
        signal.alarm(timeout)
            
        # Phần 2: Phân tích video (thử nếu còn thời gian)
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
            print(f"⚠️ Phân tích video timeout hoặc gặp lỗi: {e}")
        
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
            
        return summary, voice_over_path
    
    except TimeoutError:
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
    
    except Exception as e:
        print(f"❌ Lỗi trong quá trình xử lý: {e}")
        summary = f"Đã xảy ra lỗi trong quá trình xử lý: {e}"
        summary_path = os.path.join('output_videos', 'review_summary.txt')
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        return summary, None
        
    finally:
        # Reset alarm
        signal.alarm(0)
