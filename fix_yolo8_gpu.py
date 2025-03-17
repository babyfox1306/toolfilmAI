import os
import torch
import sys

def fix_yolo8_gpu_detection():
    """Force GPU initialization for YOLOv8"""
    print("\n===== FIXING YOLOV8 GPU DETECTION =====")
    
    # Set environment variables
    os.environ['FORCE_CUDA'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['ULTRALYTICS_DEVICE'] = 'cuda:0'  # Key fix - force Ultralytics to use CUDA explicitly
    
    # Check GPU status
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    # Check if PyTorch is compiled with CUDA
    if "+cpu" in torch.__version__:
        print("\n⚠️ ERROR: PyTorch is compiled WITHOUT CUDA support!")
        print("   Your PyTorch version is CPU-only, which cannot use GPU acceleration.")
        print("   Please run 'run_cuda_fix.bat' to reinstall PyTorch with CUDA support.")
        
        user_input = input("\nDo you want to continue anyway with CPU mode? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Exiting. Please run 'run_cuda_fix.bat' to fix the installation.")
            sys.exit(1)
    
    if cuda_available:
        # Show CUDA information
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Force initialization of CUDA
        torch.cuda.init()
        torch.cuda.empty_cache()
        
        # Create a tensor on GPU to verify it works
        try:
            x = torch.tensor([1.0, 2.0], device="cuda")
            print(f"Test tensor created on {x.device}")
            print("✅ GPU initialization successful")
            
            # Patch ultralytics YOLO class if present
            try:
                from ultralytics import YOLO
                original_init = YOLO.__init__
                
                def patched_init(self, model='yolov8n.pt', task=None):
                    original_init(self, model, task)
                    # Force model to CUDA after initialization
                    self.to('cuda')
                    print(f"✅ YOLOv8 model forced to {next(self.parameters()).device}")
                
                YOLO.__init__ = patched_init
                print("✅ Patched YOLOv8 initialization to force CUDA")
            except ImportError:
                print("⚠️ Unable to patch ultralytics YOLO - not installed")
                
        except Exception as e:
            print(f"⚠️ Error during GPU test: {e}")
    else:
        print("❌ CUDA not available through PyTorch")
        print("\n⚠️ PyTorch cannot see your GPU. This could be due to:")
        print("   1. PyTorch is CPU-only version (most likely)")
        print("   2. NVIDIA drivers are not installed correctly")
        print("   3. Your GPU is not CUDA-compatible")
        print("\n🔧 Please run 'run_cuda_fix.bat' to reinstall PyTorch with CUDA support")
    
    # Create monkey patch functions for YOLOv8
    print("\n===== CREATING MONKEY PATCHES FOR YOLOV8 =====")
    
    # Replace initialize_yolo8_model function
    def replace_initialize_yolo8_model():
        try:
            import yolo8_detection
            
            def new_initialize_yolo8_model(device="cuda", model_size="s", task="detect"):
                """Patched initialization for YOLOv8 model with forced GPU usage"""
                from ultralytics import YOLO
                
                # Force CUDA regardless of detection
                device = "cuda"
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    print("⚠️ CUDA not detected, but still forcing CUDA mode")
                
                # Get model path
                if task == "detect":
                    model_path = f"yolov8{model_size}.pt"
                elif task == "pose":
                    model_path = f"yolov8{model_size}-pose.pt"
                else:
                    raise ValueError(f"Unsupported task: {task}")
                
                # Check models directory
                import os
                models_dir = os.path.join(os.getcwd(), "models")
                if os.path.exists(os.path.join(models_dir, model_path)):
                    model_path = os.path.join(models_dir, model_path)
                    print(f"Using model from models directory: {model_path}")
                
                print(f"🔍 Initializing YOLOv8 model for {task}...")
                
                try:
                    # Force ULTRALYTICS_DEVICE 
                    os.environ['ULTRALYTICS_DEVICE'] = 'cuda:0'
                    
                    # Load model
                    model = YOLO(model_path)
                    
                    # Explicitly move to device
                    model.to(device)
                    
                    # Check device
                    model_device = next(model.parameters()).device
                    print(f"✅ YOLOv8 model loaded on: {model_device}")
                    
                    return model
                except Exception as e:
                    print(f"❌ YOLOv8 initialization error: {e}")
                    return None
            
            # Replace the function
            yolo8_detection.initialize_yolo8_model = new_initialize_yolo8_model
            print("✅ Replaced initialize_yolo8_model function")
            
            return True
        except Exception as e:
            print(f"❌ Failed to patch YOLOv8 initialization: {e}")
            return False
    
    # Apply the patches
    if replace_initialize_yolo8_model():
        print("✅ All YOLOv8 patches applied successfully")
    else:
        print("❌ Some patches failed")
    
    print("\nLoading main application with patched YOLOv8...")

if __name__ == "__main__":
    fix_yolo8_gpu_detection()
    
    # Import main module after patching
    print("\nStarting main application...")
    import main
    main.main()
