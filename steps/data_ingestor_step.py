import pandas as pd  # Importing pandas for handling data
import sys  # Importing sys to modify the system path

# Add the source directory to the system path so that we can import modules from it
sys.path.append('/content/House_prices_prediction/src')

# Import the DataLoaderFactory class for handling different data ingestion methods
from data_load import DataLoaderFactory

# Import the step decorator from ZenML to define a pipeline step
from zenml import step


@step 
def data_ingestor_step(file_path: str) -> pd.DataFrame:
    """
    Ingest data from a ZIP file using the appropriate DataIngestor.

    Args:
        file_path (str): Path to the ZIP file containing the dataset.

    Returns:
        pd.DataFrame: The loaded dataset as a Pandas DataFrame.
    """

    # Define the expected file extension (assuming ZIP files for ingestion)
    file_ex = ".zip"

    # Retrieve the appropriate data ingestion class from the factory
    data_ingestor = DataLoaderFactory.get_data(file_ex)

    # Use the ingestor to read data from the specified file path
    df = data_ingestor.ingest(file_path)

    # Return the loaded DataFrame
    return df
