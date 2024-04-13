import os

import pandas as pd

from preprocessing.preprocessing_python import PreprocessorPipeline


preprocessor = PreprocessorPipeline()

for dir, _, files in os.walk('./data/new_webscrapping/'):
    for file in files:
        df = pd.read_csv(dir+file)
        df_processed = preprocessor.process(df)
        df_processed.to_pickle('./data/new_webscrapping_clean'+file.split('.csv')[0])
    break