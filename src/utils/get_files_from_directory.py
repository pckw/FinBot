import os


def get_files_from_directory(source_directory: str) -> list:
    """
    Get a list of files from a given directory.

    Args:
        source_directory (str): The path to the directory to get the files from.

    Returns:
        list: A list of files in the directory that have a ".pdf" extension.
    """
    files_in_directory = []
    for file in os.listdir(source_directory):
        # check if file is a pdf
        if file.endswith(".pdf"):
            files_in_directory.append(file)
    return files_in_directory
