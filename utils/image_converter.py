import os
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

class ImageConverter:
    def __init__(self, input_folder, output_folder, csv_path, image_size=512):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.csv_path = csv_path
        self.image_size = image_size

    def _convert_single_dcm_to_png(self, file_name, label):
        # Check if the DICOM file exists
        dicom_path = os.path.join(self.input_folder, file_name + '.dcm',)
        if not os.path.exists(dicom_path):
            print(f"Warning: DICOM file not found for {dicom_path}")
            return

        # Read DICOM file
        dicom_data = pydicom.dcmread(dicom_path)

        # Convert DICOM to PNG
        image_array = dicom_data.pixel_array
        image = Image.fromarray(image_array)
        image = image.resize((self.image_size, self.image_size))

        # Define the output path based on train/test/val and label
        if self.index % 5 == 0:  # 20% for validation
            output_path = os.path.join(self.output_folder, f'val/{label}/{file_name[:-4]}.png')
        elif self.index % 5 == 1:  # 20% for test
            output_path = os.path.join(self.output_folder, f'test/{label}/{file_name[:-4]}.png')
        else:  # 60% for train
            output_path = os.path.join(self.output_folder, f'train/{label}/{file_name[:-4]}.png')

        # Save the image
        image.save(output_path)

    def _convert_dcm_to_png_for_index(self, index_row):
        self.index, row = index_row
        file_name = row['patientId']
        label = row['Target']
        self._convert_single_dcm_to_png(file_name, label)

    def convert_dcm_to_png_parallel(self):
        # Create output folders if they don't exist
        for folder in ['train/0/', 'train/1/', 'test/0/', 'test/1/', 'val/0/', 'val/1/']:
            os.makedirs(os.path.join(self.output_folder, folder), exist_ok=True)

        # Read CSV file
        df = pd.read_csv(self.csv_path)
        print('Total files: ', df.shape[0])

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(self._convert_dcm_to_png_for_index, df.iterrows()), total=len(df), desc="Converting images"))

if __name__ == "__main__":
    input_folder = "data\stage_2_train_images"
    output_folder = "data/rsna-pneumonia-dataset"
    csv_path = "data\stage_2_train_labels.csv"

    image_converter = ImageConverter(input_folder, output_folder, csv_path)
    image_converter.convert_dcm_to_png_parallel()
