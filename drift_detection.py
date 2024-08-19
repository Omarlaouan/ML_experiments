import pandas as pd

import alibi
from alibi_detect.cd import TabularDrift

import evidently


import warnings
warnings.filterwarnings('ignore')

def alibi_detect_drift (X_ref, X_test, p_val) : 
    '''
    Simple function to detect drift with alibi package
    '''

    # Initialise the drift detector with reference data and a p value for statistical significance
    cd = TabularDrift(x_ref=X_ref, p_val=p_val, categories_per_feature=None)
    
    # Predict if there is drift
    preds = cd.predict(X_test)

    # Format and return result
    labels = {1:'Drift detected', 0: 'No drift'}
    result = labels[preds['data']['is_drift']]
    return result