import os
import warnings
import shutil
import hashlib
from datetime import datetime
from helpers.constants import UPLOADED_FILES_DIR, HASH_FILE_PATH
from helpers.config import vectorstore

os.makedirs(UPLOADED_FILES_DIR, exist_ok=True)


# Helper functions for loading and saving file hashes
def get_file_hash(file_path):
    """Calculate MD5 hash of a file to identify duplicates"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_existing_hashes():
    """Load existing file hashes from storage"""
    if not os.path.exists(HASH_FILE_PATH):
        return {}
    
    file_hashes = {}
    try:
        with open(HASH_FILE_PATH, "r") as f:
            for line in f:
                if ":" in line:
                    hash_value, filename = line.strip().split(":", 1)
                    file_hashes[hash_value] = filename
    except Exception:
        # If there's an error reading the file, start fresh
        file_hashes = {}
    
    return file_hashes

def save_file_hash(file_hash, filename):
    """Save a new file hash to storage"""
    with open(HASH_FILE_PATH, "a+") as f:
        f.write(f"{file_hash}:{filename}\n")
        
 
def save_uploaded_files(files):
    """Save uploaded files to the designated directory and return their paths"""
    saved_paths = []
    duplicate_files = []
    existing_hashes = load_existing_hashes()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, file in enumerate(files):
        # Calculate file hash to check for duplicates
        file_hash = get_file_hash(file.name)
        
        # Check if this file has been uploaded before
        if file_hash in existing_hashes:
            duplicate_files.append(os.path.basename(file.name))
            continue
        
        # Create a unique filename with timestamp
        original_filename = os.path.basename(file.name)
        filename_without_ext, extension = os.path.splitext(original_filename)
        new_filename = f"{filename_without_ext}_{timestamp}_{i}{extension}"
        
        # Define the save path
        save_path = os.path.join(UPLOADED_FILES_DIR, new_filename)
        
        # Copy the file to the uploads directory
        shutil.copy2(file.name, save_path)
        saved_paths.append(save_path)
        
        # Save the hash to prevent future duplicates
        save_file_hash(file_hash, new_filename)
    
    return saved_paths, duplicate_files

def list_uploaded_files():
    """List all previously uploaded PDF files with metadata"""
    if not os.path.exists(UPLOADED_FILES_DIR):
        return {"data": [], "headers": ["Filename", "Size (KB)", "Upload Date"]}
    
    files_data = []
    for filename in os.listdir(UPLOADED_FILES_DIR):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(UPLOADED_FILES_DIR, filename)
            # Get file stats
            file_stats = os.stat(file_path)
            size_kb = file_stats.st_size / 1024
            upload_date = datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M")
            
            # Add as a list (row) instead of a dictionary
            files_data.append([filename, f"{size_kb:.1f}", upload_date])
    
    # Sort by upload date (newest first)
    files_data.sort(key=lambda x: x[2], reverse=True)
    
    return {"data": files_data, "headers": ["Filename", "Size (KB)", "Upload Date"]}

def remove_pdf_from_vectorstore(pdf_filename: str):
    all_docs = vectorstore.get()
    
    # Silinecek belgeleri filtreleme
    doc_ids_to_remove = [doc["id"] for doc in all_docs["documents"] if doc["metadata"].get("source") == pdf_filename]
    
    if not doc_ids_to_remove:
        print(f"⚠️ {pdf_filename} not found in vectorstore.")
        return
    
    vectorstore.delete(doc_ids_to_remove)
    vectorstore.persist()
    
    print(f"❌ {pdf_filename} removed successfully!")

