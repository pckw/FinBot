import os


def get_source_pdf_from_directory(source_directory: str) -> str:
    """
    Given a source directory, this function retrieves the path of a PDF file within the directory.

    Args:
        source_directory (str): The path to the directory containing the PDF files.

    Returns:
        str: The path of the PDF file within the directory.

    Raises:
        SystemExit: If there is not exactly one PDF file in the directory.
    """
    files = []
    for file in os.listdir(source_directory):
        if file.endswith(".pdf"):
            files.append(os.path.join(source_directory, file))
    if len(files) != 1:
        print(f"Error: There should be only one PDF file in the directory. Found {len(files)} files.")
        exit()
    return files[0]
