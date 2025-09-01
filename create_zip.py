import os
import zipfile

# Define the name of the output zip file
zip_file_name = 'archive.zip'

# Create a ZipFile object in write mode
with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Walk through all files and folders in the current directory
    for root, dirs, files in os.walk('.'):
        # Exclude directories
        dirs[:] = [d for d in dirs if d not in ['.git', '.pytest_cache', '__pycache__']]
        
        # Exclude the zip file itself from the list of files
        files[:] = [f for f in files if f != zip_file_name]

        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, os.path.relpath(file_path, '.'))

print(f"Successfully created zip file: {zip_file_name}")