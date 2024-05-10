import re

import pandas as pd


class PreprocessorPipeline:
    """
    A class for processing DataFrame data, particularly for social media content.
    It provides methods for handling missing data, casting data types, cleaning
    tweets, and normalizing user handles.

    Attributes:
    - verbose (bool): If True, outputs messages during data processing.

    Methods:
    - __init__(self, verbose: bool = False): Initializes the class with an
      option for verbose output.
    - convert_to_int(value): Converts values to integers, handling thousands
      represented as 'K'.
    - _dealing_with_na(self, df): Fills missing numeric values with zeros and
      drops rows where 'TweetText' is empty.
    - _cast_columns(self, df): Converts data types of specific DataFrame columns.
    - _cleaning_tweets(self, df, column): Cleans text by lowercasing, removing
      special characters, URLs, non-alphanumeric characters, and collapsing
      repeated characters.
    - _loosing_handle(self, df): Removes the '@' character from 'Handle' column
      and trims whitespace.
    - process(self, df): Processes the DataFrame through the cleaning functions
      and returns the cleaned DataFrame.

    Example usage:
    ```python
    # Example data
    data = {
        "Handle": ["@user1", "@user2 "],
        "TweetText": ["Hello world! http://link.co", "Great day! #coding"],
        "ReplyCount": ["five", "2"],
        "RetweetCount": [None, "3"],
        "LikeCount": ["10", None],
        "PostDate": ["2021-01-01", "2021-01-02"]
    }
    df = pd.DataFrame(data)
    
    # Initialize the PreprocessorPipeline with verbose output
    preprocessor = PreprocessorPipeline(verbose=True)
    
    # Process the data
    processed_df = preprocessor.process(df)
    print(processed_df)
    ```
    """
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    @staticmethod
    def convert_to_int(value):
        value = str(value).replace(",", '.')
        if "K" in value:
            return int(float(value.replace("K", "")) * 1000)
        return int(float(value))

    def _dealing_with_na(self, df):
        """
        Handles missing values in specific columns of a DataFrame
        to prepare the data for analysis.

        This method performs the following operations:
        - Fills missing values in 'ReplyCount', 'RetweetCount', and 'LikeCount'
          columns with zero.
        This is useful for numerical calculations where NA can lead to errors.
        - Drops rows where the 'TweetText' column is empty.
          This step removes tweets that might not contribute to
          meaningful analysis because they lack text content.

        If the 'verbose' attribute of the object is set to True,
        the method will print a message indicating that it is
        dealing with missing values.

        Parameters:
        - df (pd.DataFrame): The DataFrame in which missing values
          need to be handled.

        Returns:
        - pd.DataFrame: The DataFrame with missing values handled according to
          the specified rules.

        Raises:
        - KeyError: If the specified columns are not present in the DataFrame.

        Example:
        ```python
        data = {
            "TweetText": ["Hello world!", None, "This is a test tweet"],
            "ReplyCount": [None, 2, 3],
            "RetweetCount": [1, None, 4],
            "LikeCount": [None, 1, None]
        }
        df = pd.DataFrame(data)
        cleaned_df = instance._dealing_with_na(df)
        print(cleaned_df)
        ```
        """
        if self.verbose:
            print("---Dealing with na values...")

        # Filling 0 for int columns
        df.loc[:, ["ReplyCount", "RetweetCount", "LikeCount"]] = df[
            ["ReplyCount", "RetweetCount", "LikeCount"]
        ].fillna(0)

        # Dropping empty tweets
        df = df.dropna(subset=["TweetText"])

        return df

    def _cast_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the data types of specific columns in a DataFrame to more appropriate types for further analysis.

        This method modifies the 'PostDate', 'ReplyCount', 'RetweetCount', and 'LikeCount' columns of the DataFrame:
        - 'PostDate' is converted to a datetime object to enable time-based analysis.
        - 'ReplyCount', 'RetweetCount', and 'LikeCount' are converted to integers for numerical operations.

        If the 'verbose' attribute of the object is set to True, the method will print a message indicating the start of the type conversion process.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the columns to be cast.

        Returns:
        - pd.DataFrame: The modified DataFrame with the specified columns converted to their new data types.

        Raises:
        - ValueError: If there are any issues in converting the data types, which might occur if the data is not formatted correctly or contains invalid values that cannot be converted.

        Example:
        ```python
        data = {
            "PostDate": ["2021-01-01", "2022-01-01"],
            "ReplyCount": ["5", "10"],
            "RetweetCount": ["15", "20"],
            "LikeCount": ["25", "30"]
        }
        df = pd.DataFrame(data)
        transformed_df = instance._cast_columns(df)
        print(transformed_df.dtypes)
        ```
        """
        if self.verbose:
            print("---Changing the type of columns...")

        df.loc[:, "PostDate"] = pd.to_datetime(df["PostDate"])

        df.loc[:, "ReplyCount"] = df["ReplyCount"].apply(self.convert_to_int)
        df.loc[:, "RetweetCount"] = df["RetweetCount"].apply(self.convert_to_int)
        df.loc[:, "LikeCount"] = df["LikeCount"].apply(self.convert_to_int)

        return df

    def _cleaning_tweets(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Cleans the text content in a specified column of a DataFrame by performing multiple text normalization
        and cleaning steps. This function focuses on preparing tweet text for further natural language processing tasks.

        The operations performed include:
        - Converting all text to lowercase.
        - Removing newline characters.
        - Stripping text after a specific character ('·') if present.
        - Removing URLs.
        - Eliminating all characters that are not alphabetic, numeric,
          or spaces.
        - Collapsing multiple non-word characters into a single space and
          trimming leading/trailing spaces.
        - Reducing repeated characters in words to enhance uniformity in text.
        - Dropping duplicate entries based on the cleaned column to ensure
          uniqueness.

        If the 'verbose' attribute is True, it prints a message specifying the
        column being cleaned.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the tweet data.
        - column (str): The name of the column containing text to clean.

        Returns:
        - pd.DataFrame: The DataFrame with the text in the specified column
          cleaned.

        Raises:
        - KeyError: If the specified column is not present in the DataFrame.

        Example:
        ```python
        data = {
            "tweets": [
                "Exciting news! https://news.example.com Check it out · 2023",
                "Hurryyyy!!! Meet us there... #event",
                "Great deal at: http://deal.example.com\nDon't miss out!"
            ]
        }
        df = pd.DataFrame(data)
        cleaned_df = instance._cleaning_tweets(df, "tweets")
        print(cleaned_df)
        ```
        """
        if self.verbose:
            print(f"---Cleaning dataframe column: {column}")

        # Lowering text
        df.loc[:, column] = df[column].str.lower()

        # Deleting special characters (\n)
        df.loc[:, column] = df[column].str.replace(r"[\n]+", " ", regex=True)

        # Deleting any character before the ·
        df.loc[:, column] = df[column].apply(
            lambda tweet: tweet[tweet.find("·") + 1 :] if "·" in tweet else tweet
        )

        # Deleting URLs
        df.loc[:, column] = df[column].str.replace(r"https?://\S+", "", regex=True)

        # Deleting any character that is not an uppercase or lowercase letter, a digit, or a space
        df.loc[:, column] = df[column].str.replace(r"[^A-Za-z0-9 ]", " ", regex=True)

        # # Keeping only the text part of the tweets (if there is a need to extract after certain years)
        # df[column] = df[column].str.extract(r"(?<=2017|2018|2019|2020|2021|2022|2023|2024)(.*)")[0]

        # Splitting by word boundaries and replacing non-word characters
        df.loc[:, column] = df[column].str.replace(r"\W+", " ", regex=True).str.strip()

        # Repeating words like hurrrryyyyyy
        def rpt_repl(match):
            return match.group(1) + match.group(1)

        df.loc[:, column] = df[column].apply(
            lambda x: re.sub(r"(.)\1{1,}", rpt_repl, x) if pd.notna(x) else x
        )

        # Dropping duplicates
        df = df.drop_duplicates()

        return df

    def _loosing_handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes the '@' character from the 'Handle' column of a DataFrame. This function is typically used to
        clean up social media handle data, making it consistent and easier to analyze or display by removing
        the '@' symbol, which is commonly used in social media platforms to indicate a user handle.

        The function also trims any leading or trailing whitespace from the handle data after the '@' symbol is removed.

        If the 'verbose' attribute is True, this function prints a message indicating that handles are being cleaned.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the 'Handle' column to be cleaned.

        Returns:
        - pd.DataFrame: The DataFrame with the '@' symbol removed from the 'Handle' column.

        Raises:
        - KeyError: If the 'Handle' column does not exist in the DataFrame.

        Example:
        ```python
        data = {
            "Handle": ["@user1", "@user2 ", " @user3"]
        }
        df = pd.DataFrame(data)
        cleaned_df = instance._loosing_handle(df)
        print(cleaned_df)
        ```
        """

        if self.verbose:
            print("---Cleaning the handles...")

        df.loc[:, "Handle"] = df["Handle"].str.replace("@", "").str.strip()

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a DataFrame containing social media data by applying a
        series of cleaning and formatting functions.
        This method is a pipeline that includes handling missing values,
        casting column types, cleaning tweet texts, and
        normalizing user handles. The goal is to prepare the data for further
        analysis or use in applications where
        clean and structured data is necessary.

        The steps in the processing pipeline are:
        - `_dealing_with_na`: Handles missing values in specified columns.
        - `_cast_columns`: Converts the data types of specific columns such as 
          dates and numeric counts.
        - `_cleaning_tweets`: Cleans the 'TweetText' column by lowering case, 
          removing special characters, URLs, etc.
        - `_loosing_handle`: Removes the '@' from user handles and trims extra 
          whitespace.

        If the 'verbose' attribute is True, the method will also print the 
        first three entries of the DataFrame after
        processing to provide a quick overview of the result.

        Parameters:
        - df (pd.DataFrame): The DataFrame to be processed.

        Returns:
        - pd.DataFrame: The processed DataFrame with cleaned and formatted 
          data.

        Example:
        ```python
        data = {
            "Handle": ["@user1", "@user2 "],
            "TweetText": ["Hello world! http://link.co", "Great day! #coding"],
            "ReplyCount": ["five", "2"],
            "RetweetCount": [None, "3"],
            "LikeCount": ["10", None],
            "PostDate": ["2021-01-01", "2021-01-02"]
        }
        df = pd.DataFrame(data)
        processed_df = instance.process(df)
        print(processed_df)
        ```
        """
        df = self._dealing_with_na(df)
        df = self._cast_columns(df)
        df = self._cleaning_tweets(df, "TweetText")
        df = self._loosing_handle(df)

        if self.verbose:
            print("---Here is the result example :) ")
            print(df.head(3))

        return df


if __name__ == "__main__":
    df = pd.read_csv("./../data/new_webscrapping/webscraped_stora_enso.csv")
    pp = PreprocessorPipeline(verbose=True)
    cleaned_df = pp.process(df)
    print(cleaned_df)
