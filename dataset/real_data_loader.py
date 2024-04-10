import pandas as pd
from configs.RealData import RealData
config = RealData()

def load_df(
    df_path:str, 
    is_cog_opperation:bool,
    sampling_period:int=config.sampling_period, 
    smothing_period:int=config.smoothing_period, 
    min_smoothing_period:int=config.smoothing_period,
    cols_of_interest:list=config.cols_on_interrest,
    reset_index:bool=config.reset_index):

    _df = pd.read_hdf(df_path, key="data")
    _df = _df.rename(columns={"Hot metal temperature (sampling date per sample)":"Hot metal temperature (per sample)"})
    
    # in case its is Ternium Dataset
    _df = _df.rename(columns={"Validation_1min.Gen_HM_Temp_Per_Sample": 'Hot metal temperature (per sample)',
                              "Validation_OLM.OLMB_Coke_Rate_Dry_SP":"OLMB cycle dry coke rate - weight SP"})
    
    _df = _df.drop(columns=['BF status (0=Normal; -1=Slow down; -2=Out of operation)'])

    _df = _df.rolling(smothing_period, min_smoothing_period, win_type="blackman").mean()
    t = pd.date_range(_df.index[0], _df.index[-1], freq=f'{sampling_period}T')
    _df = _df.reindex(t)
    
    if not is_cog_opperation:
        _df['RXT_COG_Rate_Current'] = _df['RXT_COG_Rate_Current'].fillna(0)

    if reset_index:
        _df = _df.reset_index()

    _df = _df[cols_of_interest]
    return _df


