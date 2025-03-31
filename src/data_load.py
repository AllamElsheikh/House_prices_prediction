import os 
import zipfile
import argparse
from abc import ABC, abstractmethod
import pandas as pd

class DataLoading(ABC):
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Abstract method for loading data from a given file."""
        pass

class ZipDataLoader(DataLoading):
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Extracts data from a .zip file and returns the content as a pandas DataFrame."""
        
        if not file_path.endswith(".zip"):
            raise ValueError("The file is not a zip file")
        
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")
        
        extracted_files = os.listdir("extracted_data")
        if len(extracted_files) == 0:
            raise FileNotFoundError("No files found in the extracted directory")
        if len(extracted_files) > 1:
            raise ValueError("More than one file found in the extracted directory")
        
        data_path = os.path.join("extracted_data", extracted_files[0])
        df = pd.read_csv(data_path)
        return df

class DataLoaderFactory:
    @staticmethod
    def get_data(file_extension: str) -> DataLoading:
        if file_extension == ".zip":
            return ZipDataLoader()
        else:
            raise ValueError("Unsupported file extension")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and extract data from a ZIP file.")
    parser.add_argument("file_path", type=str, help="Path to the zip file.")
    args = parser.parse_args()
    
    try:
        loader = DataLoaderFactory.get_data(".zip")
        df = loader.load_data(args.file_path)
        print("Data Loaded Successfully:")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")
