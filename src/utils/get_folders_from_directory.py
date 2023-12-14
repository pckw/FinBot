import os

def get_folders_from_directory(source_directory: str):
    folders = []
    for file in os.listdir(source_directory):
        if os.path.isdir(os.path.join(source_directory, file)):
            folders.append(os.path.join(source_directory, file))
    return folders