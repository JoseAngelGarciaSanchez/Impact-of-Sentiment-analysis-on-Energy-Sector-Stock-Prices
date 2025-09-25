import os

import pandas as pd

from model.sentimental_model import SentimentalAnalysisModel
from model.parameters import concatenated_info
from preprocessing.preprocessing import PreprocessorPipeline


class LaunchSystem:
    def __init__(self) -> None:
        self.preprocessor = PreprocessorPipeline(verbose=True)
        self.sentimental_model = SentimentalAnalysisModel()

    def _processing_tweets(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = self.preprocessor.process(df)
        return df_processed

    def _sentimental_analysis(self, df_processed: pd.DataFrame) -> pd.DataFrame:
        df_predicted = self.sentimental_model.predict(df_processed)
        return df_predicted

    def _concatenated_prediction(self, dir: str | None = None):
        """
        dir: str | None = None
        ----example: "./example/path"
        """
        concatenated_prediction = pd.DataFrame()
        for directoty, _, files in os.walk(dir):
            files = [file for file in files if file.startswith("webscraped_")]
            for file in files:
                df = pd.read_csv(directoty + file)
                df["company"] = concatenated_info[file]
                concatenated_prediction = pd.concat(
                    [concatenated_prediction, df], ignore_index=True
                )
            concatenated_prediction.to_csv(
                f"{directoty}concatenated_prediction.csv", index=False
            )
            break

    def company_process(self, company_name: str | None = None):
        """
        company_name: str | None = 'webscraped_stora_enso'
        ----example: "webscraped_stora_enso"
        """
        df = pd.read_csv(f"./../data/webscrapped/raw/twitter/{company_name}.csv")
        df_processed = self._processing_tweets(df=df)
        df_predicted = self.sentimental_model.predict(df_processed)

        df_processed.to_pickle(f"./../data/webscrapped/processed/twitter/{company_name}.pkl")
        df_predicted.to_csv(
            f"./../data/webscrapped/predicted/twitter/{company_name}.csv",
            index=False,
        )

        self._concatenated_prediction(dir="./../data/webscrapped/predicted/twitter/")

    def directory_process(self, dir: str | None = None):
        """
        dir: str | None = None
        ----example: "./data/new_webscrapping/"
        """
        for directory, _, files in os.walk(dir):
            for file in files:
                df = pd.read_csv(directory + file)
                df_processed = self.preprocessor.process(df)
                df_predicted = self.sentimental_model.predict(df_processed)

                df_processed.to_pickle(
                    f'./../data/new_webscrapping_clean/{file.split(".csv")[0]}.pkl'
                )
                df_predicted.to_csv(
                    f'./../data/new_webscrapping_predicted/{file.split(".csv")[0]}.csv',
                    index=False,
                )
            break

        self._concatenated_prediction(dir="./../data/new_webscrapping_predicted/")


if __name__ == "__main__":
    ls = LaunchSystem()
    ls.company_process("webscraped_totalenergies_se")