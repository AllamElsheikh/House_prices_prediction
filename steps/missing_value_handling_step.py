import pandas as pd
import sys
from zenml import step

sys.path.append('/content/House_prices_prediction/src') 
from missing_value_handdling import ( 
    DropMissingValues ,
    FillMissingValues 
    )




@step 
def missing_values_handling_step(
    df : pd.DataFrame , strategy  :str =  "mean" ) -> pd.DataFrame : 
    
    if strategy == "mean":
        df_cleaned = FillMissingValues(method = "mean").handling(df)
    elif strategy == "median":
        df_cleaned = FillMissingValues(method = "median").handling(df)
    elif strategy == "mode":
        df_cleaned = FillMissingValues(method = "mode").handling(df)
    elif strategy == "drop":
        df_cleaned = DropMissingValues().handling(df)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'.")
    return df_cleaned