import os
import sys
import torch
import subprocess
import importlib
import shutil
from pathlib import Path

# TryExcept class mà YOLOv5 đang tìm kiếm nhưng không tìm thấy
class TryExcept:
    """
    Class tương thích với YOLOv5 để giải quyết lỗi import
    """
    def __init__(self, *args, **kwargs):
        pass
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Thêm vào sys.modules để đảm bảo có thể import
# Tạo module utils nếu chưa có
if 'utils' not in sys.modules:
    sys.modules['utils'] = type('', (), {})()
    
# Thêm TryExcept vào module utils
sys.modules['utils'].TryExcept = TryExcept

def fix_yolo5_imports():
    """Sửa lỗi import YOLOv5"""
    try:
        # Cách 1: Kiểm tra nếu YOLOv5 đã được cài qua pip
        try:
            import yolov5
            print("✅ Sử dụng YOLOv5 từ pip package")
            return True
        except ImportError:
            print("⚠️ YOLOv5 chưa được cài qua pip, thử phương pháp khác...")
            
        # Cách 2: Sử dụng thư mục YOLOv5 local
        yolov5_dir = Path("yolov5")
        if not yolov5_dir.exists():
            print("🔍 Thư mục YOLOv5 không tồn tại, đang tải về...")
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/ultralytics/yolov5", str(yolov5_dir)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
                )
                print("✅ Đã tải YOLOv5 từ GitHub")
            except Exception as e:
                print(f"⚠️ Không thể clone repository: {e}")
                print("Đang tải ZIP file...")
                
                # Dùng phương pháp tải ZIP thay thế
                import urllib.request
                import zipfile
                try:
                    zip_url = "https://github.com/ultralytics/yolov5/archive/refs/heads/master.zip"
                    zip_path = "yolov5-master.zip"
                    urllib.request.urlretrieve(zip_url, zip_path)
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(".")
                        
                    if os.path.exists("yolov5-master"):
                        os.rename("yolov5-master", "yolov5")
                        
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                        
                    print("✅ Đã tải và giải nén YOLOv5")
                except Exception as e2:
                    print(f"❌ Không thể tải ZIP: {e2}")
                    # Thử cài đặt qua pip làm phương án cuối
                    try:
                        subprocess.run(
                            [sys.executable, "-m", "pip", "install", "yolov5"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
                        )
                        print("✅ Đã cài đặt YOLOv5 qua pip")
                        import yolov5
                        return True
                    except Exception as e3:
                        print(f"❌ Không thể cài đặt YOLOv5: {e3}")
                        return False
        
        # Thêm thư mục YOLOv5 vào sys.path
        yolov5_path = str(yolov5_dir.absolute())
        if yolov5_path not in sys.path:
            sys.path.insert(0, yolov5_path)
            print(f"✅ Đã thêm {yolov5_path} vào sys.path")
        
        # Thêm các module giả để thay thế
        try:
            # Kiểm tra xem có thể import được các module cần thiết không
            # CHÚ Ý: Phải import đầy đủ từ yolov5
            from yolov5.models import common
            from yolov5.utils import torch_utils
            print("✅ Đã import thành công modules từ YOLOv5 local")
            return True
        except ImportError as e:
            print(f"⚠️ Vẫn còn lỗi import: {e}")
            # Tạo các module ảo
            sys.modules['models'] = type('', (), {})()
            sys.modules['models.common'] = type('', (), {})()
            sys.modules['utils.torch_utils'] = type('', (), {})()
            print("⚠️ Đã tạo các module ảo, có thể gặp lỗi khi chạy")
            return False
    except Exception as e:
        print(f"❌ Lỗi không mong đợi: {e}")
        return False

def initialize_yolo5_model(weights="yolov5s.pt", device="cuda"):
    """Khởi tạo model YOLOv5"""
    print(f"🔍 Đang khởi tạo model YOLOv5 từ {weights}...")
    
    # Kiểm tra file weights
    weights_path = weights
    
    # Kiểm tra trong thư mục models trước
    models_dir = Path("models")
    if models_dir.exists():
        model_in_dir = models_dir / Path(weights).name
        if model_in_dir.exists():
            print(f"✅ Đã tìm thấy model trong thư mục models: {model_in_dir}")
            weights_path = str(model_in_dir)
    
    # Phương pháp 1: Sử dụng pip package (ưu tiên)
    try:
        import yolov5
        model = yolov5.load(weights_path, device=device)
        print(f"✅ Đã tải model thành công qua pip package")
        return model
    except ImportError:
        print("⚠️ Không tìm thấy package yolov5, đang thử phương pháp khác...")
    except Exception as e:
        print(f"⚠️ Lỗi khi tải model qua pip: {e}")
    
    # Phương pháp 2: Sử dụng torch.hub
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, device=device)
        print(f"✅ Đã tải model thành công qua torch.hub")
        return model
    except Exception as e:
        print(f"⚠️ Lỗi khi tải model qua torch.hub: {e}")
    
    # Phương pháp 3: Import trực tiếp
    try:
        # Đảm bảo đường dẫn được thiết lập
        fix_yolo5_imports()
        
        # Thử import các module cần thiết từ thư mục local
        try:
            from yolov5.models.common import AutoShape
            from yolov5.models.experimental import attempt_load
            from yolov5.utils.torch_utils import select_device
            
            # Khởi tạo model
            device_obj = select_device(device)
            model = attempt_load(weights_path, device=device_obj)
            model = AutoShape(model)
            
            print(f"✅ Đã tải model thành công từ local YOLOv5")
            return model
        except ImportError as e:
            print(f"⚠️ Lỗi import từ local YOLOv5: {e}")
            
        # Thử cài đặt và import lại
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "yolov5"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            import yolov5
            model = yolov5.load(weights_path, device=device)
            print(f"✅ Đã tải model sau khi cài đặt yolov5")
            return model
        except Exception as e:
            print(f"❌ Không thể tải model sau nhiều nỗ lực: {e}")
    
    except Exception as e:
        print(f"❌ Lỗi không mong đợi: {e}")
    
    return None

def detect_objects_with_yolo5(image, model=None, classes=None, conf=0.25):
    """
    Phát hiện đối tượng trong hình ảnh sử dụng YOLOv5
    
    Args:
        image: Numpy array chứa ảnh (BGR)
        model: YOLOv5 model (nếu None, sẽ tự động tải)
        classes: Danh sách các class cần phát hiện (None để phát hiện tất cả)
        conf: Ngưỡng tin cậy
    
    Returns:
        List of [x1, y1, x2, y2, confidence, class_id]
    """
    if model is None:
        model = initialize_yolo5_model()
        if model is None:
            return []
    
    # Thiết lập các thông số
    model.conf = conf
    model.classes = classes
    
    # Thực hiện dự đoán
    results = model(image)
    
    # Convert kết quả thành format dễ sử dụng
    detections = []
    
    if hasattr(results, 'xyxy'):
        # YOLOv5 from torch hub
        boxes = results.xyxy[0]  # Tensor format
        for box in boxes:
            # Format: [x1, y1, x2, y2, confidence, class]
            detections.append(box.cpu().numpy().tolist())
    else:
        # YOLOv5 from direct import
        try:
            boxes = results.pandas().xyxy[0]  # DataFrame format
            for _, box in boxes.iterrows():
                detection = [
                    box['xmin'], box['ymin'], box['xmax'], box['ymax'],
                    box['confidence'], box['class']
                ]
                detections.append(detection)
        except Exception:
            try:
                # Try another format
                boxes = results.pred[0]  # Tensor format
                for box in boxes:
                    detections.append(box.cpu().numpy().tolist())
            except Exception as e:
                print(f"❌ Lỗi khi xử lý kết quả YOLOv5: {e}")
    
    return detections
