#unzip the file
import zipfile

def unzip_file(filepath: str, to: str):
    zip_ref = zipfile.ZipFile(filepath, 'r')
    zip_ref.extractall(to)
    zip_ref.close()

unzip_file(filepath=r"data\rsna-pneumonia-detection-challenge.zip", to="data")