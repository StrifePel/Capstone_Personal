import os

def check_access():
    paths = [
        './data/train',
        './data/val',
        './data/test'
    ]
    
    print("Checking directory access...")
    for path in paths:
        print(f"\nChecking {path}")
        
        # Check if directory exists
        if os.path.exists(path):
            print(f"✓ Directory exists")
            
            # List all files
            try:
                files = os.listdir(path)
                print(f"✓ Can list directory contents. Found {len(files)} files:")
                for file in files:
                    print(f"  - {file}")
                    
                    # Try to open one file
                    try:
                        with open(os.path.join(path, file), 'rb') as f:
                            f.read(1024)  # Try to read first 1KB
                            print(f"  ✓ Can read file: {file}")
                    except Exception as e:
                        print(f"  ✗ Cannot read file {file}: {str(e)}")
            except Exception as e:
                print(f"✗ Cannot list directory contents: {str(e)}")
        else:
            print(f"✗ Directory does not exist")

if __name__ == "__main__":
    check_access()