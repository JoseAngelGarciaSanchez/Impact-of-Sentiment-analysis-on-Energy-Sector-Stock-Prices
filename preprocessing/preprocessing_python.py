import re
import sys

import pandas as pd

class PreprocessorPipeline:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
    
    def _dealing_with_na(self, df):
        """
        Here, we're dealing with NA values. 
        For the int columns, we're filling the NAs with 0
        and we're dropping the empty tweets.
        """
        if self.verbose:
            print("---Dealing with na values...")

        # Filling 0 for int columns
        df.loc[:, ['ReplyCount', 'RetweetCount', 'LikeCount']] = df[['ReplyCount', 'RetweetCount', 'LikeCount']].fillna(0)

        # Dropping empty tweets
        df = df.dropna(subset=["TweetText"])

        return df

    def _cast_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        To do: format counts to int
        """
        if self.verbose:
            print('---Changing the type of columns...')

        df['PostDate'] = pd.to_datetime(df['PostDate'])

        return df

    def _cleaning_tweets(self, df, column):
        """
        In this function, we're cleaning the text of the tweets. 
        We're trimming all the special characters, the links,
        and keeping only the text part.
        """
        if self.verbose:
            print(f'---Cleaning dataframe column: {column}')

        # Lowering text
        df[column] = df[column].str.lower()

        # Deleting special characters (\n)
        df[column] = df[column].str.replace(r"[\n]+", " ", regex=True)
        
        # Deleting any character before the ·
        df[column] = df[column].apply(lambda tweet: tweet[tweet.find('·') + 1:] if '·' in tweet else tweet)

        # Deleting URLs
        df[column] = df[column].str.replace(r"https?://\S+", "", regex=True)

        # Deleting any character that is not an uppercase or lowercase letter, a digit, or a space
        df[column] = df[column].str.replace(r"[^A-Za-z0-9 ]", " ", regex=True)

        # # Keeping only the text part of the tweets (if there is a need to extract after certain years)
        # df[column] = df[column].str.extract(r"(?<=2017|2018|2019|2020|2021|2022|2023|2024)(.*)")[0]

        # Splitting by word boundaries and replacing non-word characters
        df[column] = df[column].str.replace(r"\W+", " ", regex=True).str.strip()

        # Repeating words like hurrrryyyyyy
        def rpt_repl(match):
            return match.group(1) + match.group(1)
        df[column] = df[column].apply(lambda x: re.sub(r"(.)\1{1,}", rpt_repl, x) if pd.notna(x) else x)

        # Dropping duplicates
        df = df.drop_duplicates(subset=[column])

        return df
    
    def _loosing_handle(self, df):
        """This function looses the @ for the handles."""
        
        if self.verbose:
            print("---Cleaning the handles...")

        df.loc[:, 'Handle'] = df['Handle'].str.replace('@', '').str.strip()

        return df
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        print(df.shape)
        df = self._cast_columns(df)
        df = self._cleaning_tweets(df, "TweetText")
        df = self._loosing_handle(df)
        df = self._dealing_with_na(df)
        print(df.shape)

        if self.verbose:
            print("Here is the result :) ")
            # print(df.head(3))

        return df
    

if __name__ == '__main__':
    df = pd.read_csv('./../data/new_webscrapping/webscraped_bp_plc.csv')
    pp = PreprocessorPipeline(verbose=False)
    cleaned_df = pp.process(df)