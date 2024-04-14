import os

import pandas as pd

from preprocessing.preprocessing import PreprocessorPipeline
from model.model import SentimentalAnalysisModel


preprocessor = PreprocessorPipeline()
sentimental_model = SentimentalAnalysisModel()


for dir, _, files in os.walk('./data/new_webscrapping/'):
    for file in files:
        df = pd.read_csv(dir+file)
        df_processed = preprocessor.process(df)
        df_predicted = sentimental_model.predict(df_processed)

        # Saving transformations
        df_processed.to_pickle(f'./data/new_webscrapping_clean/{file.split(".csv")[0]}.pkl')
        df_predicted.to_csv(f'./data/new_webscrapping_predicted/{file.split(".csv")[0]}.pkl', index=False)

    break
