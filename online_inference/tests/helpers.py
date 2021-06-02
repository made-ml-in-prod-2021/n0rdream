import numpy as np
import pandas as pd


def generate_fake_df(columns, size=10, target=None):
    df = pd.DataFrame()
    for col in columns:
        if col in ['sex', 'fbs', 'exang']:
            df[col] = np.random.randint(0, 2, size)
        elif col == 'age':
            df[col] = np.random.randint(29, 78, size)
        elif col == 'cp':
            df[col] = np.random.randint(0, 4, size)
        elif col == 'trestbps':
            df[col] = np.random.randint(94, 201, size)
        elif col == 'chol':
            df[col] = np.random.randint(126, 565, size)
        elif col == 'restecg':
            df[col] = np.random.randint(0, 3, size)
        elif col == 'thalach':
            df[col] = np.random.randint(71, 203, size)
        elif col == 'oldpeak':
            df[col] = np.random.randint(0, 63, size) / 10
        elif col == 'slope':
            df[col] = np.random.randint(0, 3, size)
        elif col == 'ca':
            df[col] = np.random.randint(0, 5, size)
        elif col == 'thal':
            df[col] = np.random.randint(0, 4, size)
    if target is not None:
        df[target] = np.random.randint(0, 2, size)
    return df
