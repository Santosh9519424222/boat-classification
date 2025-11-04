"""
Cleanup script to remove unnecessary files from the project.
Run this before pushing to GitHub.
"""
import os
import shutil

def remove_if_exists(path):
    """Remove file or directory if it exists"""
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"✅ Removed directory: {path}")
        else:
            os.remove(path)
            print(f"✅ Removed file: {path}")
    else:
        print(f"⚠️  Path not found: {path}")

def main():
    print("🧹 Starting cleanup process...")
    print("=" * 70)
    
    # Remove templates folder (duplicate of frontend)
    remove_if_exists('./templates')
    
    # Remove __pycache__ directories
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            remove_if_exists(os.path.join(root, '__pycache__'))
    
    # Remove .pyc files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                remove_if_exists(os.path.join(root, file))
    
    # Remove old model files if they exist
    old_models = [
        'boat_classifier_efficientnetb3.h5',
        'boat_classifier_custom_cnn.h5'
    ]
    for model in old_models:
        remove_if_exists(model)
    
    print("=" * 70)
    print("✅ Cleanup completed!")
    print("\nRemaining project structure:")
    print("  - boat-classification.ipynb (main training notebook)")
    print("  - backend/ (Flask API)")
    print("  - frontend/ (Web interface)")
    print("  - data/ (train/validation/test datasets)")
    print("  - requirements.txt (Python dependencies)")
    print("  - README.md (Project documentation)")
    print("  - .gitignore (Git ignore rules)")

if __name__ == '__main__':
    main()
