import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

def BasuEstimate(y,VARpen=None,VARMApen=None):
    # Wilms & Basu
    bigtime = importr('bigtime')
    sparseVARMA = bigtime.sparseVARMA
    sparseVAR = bigtime.sparseVAR
    #converting it into r object for passing into r function
    y_pd = pd.DataFrame(y.T)
    with localconverter(ro.default_converter + pandas2ri.converter):
        y_r = ro.conversion.py2rpy(y_pd)
    base = importr('base')
    as_matrix = base.as_matrix
    y_r = as_matrix(y_r)
    #Invoking the R function and getting the result
    if VARMApen is not None:
        df_result_r = sparseVARMA(y_r,VARMAselection = "cv",VARMApen=VARMApen,VARpen=VARpen)
    else:
        df_result_r = sparseVAR(y_r,selection = "cv",VARpen="L1")

    directforecast = bigtime.directforecast
    y_forecast_r = directforecast(df_result_r)
    with localconverter(ro.default_converter + pandas2ri.converter):
        y_forecast = ro.conversion.rpy2py(y_forecast_r)

    

    return y_forecast
    