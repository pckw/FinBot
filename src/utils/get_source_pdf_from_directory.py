import os

def get_source_pdf_from_directory(source_directory: str):
    files = []
    for file in os.listdir(source_directory):
        if file.endswith(".pdf"):
            files.append(os.path.join(source_directory, file))
    if len(files) != 1:
        print(f"Error: There should be only one PDF file in the directory. Found {len(files)} files.")
        exit()
    return files[0]
