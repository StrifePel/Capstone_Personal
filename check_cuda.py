import torch
import sys
import platform

def check_cuda():
    print("\n=== System Information ===")
    print(f"Python Version: {sys.version}")
    print(f"Operating System: {platform.system()} {platform.version()}")
    
    print("\n=== PyTorch Setup ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Get current GPU memory usage
        print("\n=== GPU Memory Usage ===")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        # Test CUDA with tensor operations
        print("\n=== CUDA Tensor Test ===")
        try:
            # Create a test tensor
            x = torch.rand(1000, 1000)
            print(f"Test tensor created on: {x.device}")
            
            # Move to GPU and perform operations
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            x = x.cuda()
            y = x @ x  # Matrix multiplication
            end_event.record()
            
            # Waits for everything to finish running
            torch.cuda.synchronize()
            
            print(f"Tensor moved to: {x.device}")
            print(f"Matrix multiplication time: {start_event.elapsed_time(end_event):.2f} ms")
            print("CUDA operations completed successfully!")
            
        except Exception as e:
            print(f"Error during CUDA tensor test: {str(e)}")
    else:
        print("\n=== CUDA Installation Guide ===")
        print("CUDA is not available. To enable CUDA support:")
        print("1. Install NVIDIA GPU drivers")
        print("2. Install CUDA Toolkit")
        print("3. Reinstall PyTorch with CUDA support using:")
        print("   pip3 uninstall torch torchvision torchaudio")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    check_cuda()