import os

import pandas as pd

from model.model import SentimentalAnalysisModel
from model.parameters import concatenated_info
from preprocessing.preprocessing import PreprocessorPipeline


preprocessor = PreprocessorPipeline()
sentimental_model = SentimentalAnalysisModel()

for dir, _, files in os.walk("./data/new_webscrapping/"):
    for file in files:
        df = pd.read_csv(dir + file)
        df_processed = preprocessor.process(df)
        df_predicted = sentimental_model.predict(df_processed)

        # Saving transformations
        df_processed.to_pickle(
            f'./data/new_webscrapping_clean/{file.split(".csv")[0]}.pkl'
        )
        df_predicted.to_csv(
            f'./data/new_webscrapping_predicted/{file.split(".csv")[0]}.csv',
            index=False,
        )
    break

concatenated_prediction = pd.DataFrame()
for dir, _, files in os.walk("./data/new_webscrapping_predicted/"):
    for file in files:
        df = pd.read_csv(dir + file)
        df["company"] = concatenated_info[file]
        concatenated_prediction = pd.concat(
            [concatenated_prediction, df], ignore_index=True
        )
    concatenated_prediction.to_csv(f"{dir}concatenated_prediction.csv", index=False)
    break
