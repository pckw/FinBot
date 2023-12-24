import os


def get_files_from_directory(source_directory: str):
    files_in_directory = []
    for file in os.listdir(source_directory):
        # check if file is a pdf
        if file.endswith(".pdf"):
            files_in_directory.append(file)
    return files_in_directory
