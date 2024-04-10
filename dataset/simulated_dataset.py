import pandas as pd
import numpy as np

from configs.SimulatedData import Proposed

config = Proposed()

def load_df(
    df_path:str, 
    cols_of_interest:list=config.cols_on_interrest):
    
    _df = pd.read_hdf(df_path, key="data")
    _df = _df[cols_of_interest]
    return _df
