import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm

from parameters import company_to_stock_dict


np.random.seed(42)
pd.options.mode.chained_assignment = None


class Preprocessing:
    def __init__(self) -> None:
        pass

    def process_analysed_tweets(self, df: pd.DataFrame):
        # Format PostDate to datetime
        df["PostDate"] = pd.to_datetime(df["PostDate"])

        # drop rows with NaN values in the "PostDate" column
        df.dropna(subset=["PostDate"], inplace=True)

        # add a column for the year and month
        df["day"] = df["PostDate"].dt.day
        df["month"] = df["PostDate"].dt.month
        df["year"] = df["PostDate"].dt.year

        # formatting sentiments
        df["sentiment"] = df["sentiment"].map({"Bullish": 1, "Bearish": -1})
        df["sentiment_base"] = df["sentiment_base"].map(
            {"positive": 1, "neutral": 0, "negative": -1}
        )
        return df

    def process_returns(self, df: pd.DataFrame):
        df = df / 100

        # Week-end
        nan_mask = df.isna().all(axis=1)
        df = df.loc[~nan_mask, :]
        return df

    def process(self, analysed_tweets: pd.DataFrame, returns: pd.DataFrame):
        analysed_tweets = self.process_analysed_tweets(analysed_tweets)
        returns = self.process_returns(returns)

        return analysed_tweets, returns


class DailyModelEvaluation:
    def __init__(self, analysed_tweets: pd.DataFrame, returns: pd.DataFrame) -> None:
        self.analysed_tweets = analysed_tweets
        self.df_returns = returns

    def _correlation_by_company(
        self,
    ):
        """
        by : day, month, year
        """

        self.positive_daily_ratios = self.__group_model_results(df=self.analysed_tweets)

        self.adjusted_returns = self.__adjust_returns_with_company_names()

    def __group_model_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Grouping model results

        by: str = ["year", "month", "day", "company"]
        """
        # group the data by year and month
        grouped = df.groupby(["year", "month", "day", "company"])

        # count the number of positive and negative tweets for each year and month
        sentiment_positive_tweets_by_day = grouped["sentiment"].apply(
            lambda x: (x == 1).sum()
        )
        sentiment_negative_tweets_by_day = grouped["sentiment"].apply(
            lambda x: (x == -1).sum()
        )
        sentiment_base_positive_tweets_by_day = grouped["sentiment_base"].apply(
            lambda x: (x == 1).sum()
        )
        sentiment_base_neutral_tweets_by_day = grouped["sentiment_base"].apply(
            lambda x: (x == 0).sum()
        )
        sentiment_base_negative_tweets_by_day = grouped["sentiment_base"].apply(
            lambda x: (x == -1).sum()
        )

        # calculate the ratio of positive and negative tweets for each year and month
        positive_ratios_by_day = (
            sentiment_base_positive_tweets_by_day + sentiment_positive_tweets_by_day
        ) / (
            sentiment_positive_tweets_by_day
            + sentiment_negative_tweets_by_day
            + sentiment_base_positive_tweets_by_day
            + sentiment_base_neutral_tweets_by_day
            + sentiment_base_negative_tweets_by_day
        )

        # formatting
        positive_ratios_by_day = positive_ratios_by_day.reset_index()
        positive_ratios_by_day.rename(columns={0: "positive_ratio"}, inplace=True)
        positive_ratios_by_day["yearmonthday"] = (
            positive_ratios_by_day["year"].astype(str)
            + "-"
            + positive_ratios_by_day["month"].astype(str).str.zfill(2)
            + "-"
            + positive_ratios_by_day["day"].astype(str).str.zfill(2)
        )

        return positive_ratios_by_day

    def __adjust_returns_with_company_names(self):
        """
        Adapt returns names to company names
        """

        selected_columns = [
            value
            for value in company_to_stock_dict.values()
            if value in self.df_returns.columns
        ]
        selected_returns = self.df_returns.loc[:, selected_columns]

        selected_returns.rename(
            columns={v: k for k, v in company_to_stock_dict.items()}, inplace=True
        )
        selected_returns = selected_returns.reset_index().rename(
            columns={"index": "date"}
        )

        return selected_returns

    def short_or_long(self):
        """new dataframe with buy or sell at t"""
        # print(self.positive_daily_ratios)
        positive_ratios_by_day = self.positive_daily_ratios.copy()

        # positive_ratios_by_day = self.__group_model_results(self.analysed_tweets)
        date_index = positive_ratios_by_day["yearmonthday"].unique().tolist()
        companies = positive_ratios_by_day["company"].unique().tolist()

        self.shortlongdf = pd.DataFrame(index=date_index)

        for company in companies:
            # selection of the company
            mask = positive_ratios_by_day["company"] == company
            stock_ratios = positive_ratios_by_day.loc[mask, :]

            # selection of the period
            stock_ratios.loc[:, "positive_ratio_shifted"] = stock_ratios[
                "positive_ratio"
            ].shift(1)

            stock_ratios.loc[:, "buy_or_sell"] = (
                stock_ratios["positive_ratio_shifted"] - 0.5
            ) * 2

            # STOCK_RATIONS BETWEEN -1 and 1
            stock_ratios.set_index("yearmonthday", inplace=True)

            self.stock_ratios = stock_ratios
            # saving info in a dataframe
            self.shortlongdf[company] = stock_ratios["buy_or_sell"]

    def evaluate_model_accuracy(self):
        self.adjusted_returns.index = pd.to_datetime(self.adjusted_returns["date"])
        self.shortlongdf.index = pd.to_datetime(self.shortlongdf.index)

        evaluation_df = self.shortlongdf.join(
            self.adjusted_returns, how="inner", lsuffix="_buysell", rsuffix="_market"
        )

        accuracy_metrics = {}

        def prediction_matches(signal, market_return):

            if signal > 0.5 and market_return > 0:  # strong positive signal
                return True
            elif signal < -0.5 and market_return < 0:  # strong negative signal
                return True
            elif (
                -0.5 <= signal <= 0.5 and -0.05 <= market_return <= 0.05
            ):  # neutral signal
                return True
            else:
                return False

        for index, row in evaluation_df.iterrows():
            for column in evaluation_df.columns:
                if "_buysell" in column:
                    company_name = column.split("_buysell")[0]
                    market_column = company_name + "_market"

                    if market_column in evaluation_df.columns:
                        if company_name not in accuracy_metrics:
                            accuracy_metrics[company_name] = {
                                "correct_predictions": 0,
                                "total_signals": 0,
                            }

                        accuracy_metrics[company_name]["total_signals"] += 1

                        if prediction_matches(row[column], row[market_column]):
                            accuracy_metrics[company_name]["correct_predictions"] += 1

        return pd.DataFrame.from_dict(
            {
                stock: {
                    "Accuracy (%)": (
                        (metrics["correct_predictions"] / metrics["total_signals"])
                        * 100
                        if metrics["total_signals"] > 0
                        else 0
                    )
                }
                for stock, metrics in accuracy_metrics.items()
            },
            orient="index",
        )

    def compute_signal_market_correlation(self):
        self.adjusted_returns.index = pd.to_datetime(self.adjusted_returns.index)
        self.shortlongdf.index = pd.to_datetime(self.shortlongdf.index)

        evaluation_df = self.shortlongdf.join(
            self.adjusted_returns, how="inner", lsuffix="_signal", rsuffix="_market"
        )

        correlation_results = {}

        for signal_column in evaluation_df.columns:
            if "_signal" in signal_column:
                base_name = signal_column.split("_signal")[0]
                market_column = base_name + "_market"

                if market_column in evaluation_df.columns:
                    
                    clean_df = evaluation_df[[signal_column, market_column]].dropna()
                    print(evaluation_df,clean_df)
                    if not clean_df.empty and len(clean_df.dropna()) >= 2:  # Check if there's enough data
                        signal_data = clean_df[signal_column]
                        market_data = clean_df[market_column]

                        # Compute correlation
                        pearson_corr, p_value = pearsonr(signal_data, market_data)
                        corr = sm.tsa.stattools.ccf(signal_data, market_data, adjusted=False)

                        # Determine significance
                        #significance = "***" if p_value and p_value < 0.05 else ""
                        formatted_correlation = f"{corr}" if corr is not None else "N/A"

                        # Store both correlation and significance under the same key
                        correlation_results[base_name] = [formatted_correlation, pearson_corr, p_value]

        # Create DataFrame with appropriate column names
        return pd.DataFrame.from_dict(
            correlation_results, orient="index", columns=["Cross-Correlation", "Pearson Correlation (need to stationarize)", "P-value"]
        )

    def save_results_to_excel(self, save_path):
        with pd.ExcelWriter(f"{save_path}daily_model_results.xlsx") as writer:
            self.evaluate_model_accuracy().to_excel(writer, sheet_name="Model Accuracy")
            self.compute_signal_market_correlation().to_excel(
                writer, sheet_name="Signal Market Correlation"
            )

        print("Results saved to daily_model_results.xlsx.")

    def visualize_courbe(self, save_path):

        os.makedirs(f"{save_path}correlation_curves/", exist_ok=True)

        evaluation_df = self.shortlongdf.join(
            self.adjusted_returns, how="inner", lsuffix="_buysell", rsuffix="_market"
        )

        def moving_average(data, window_size):
            return data.rolling(window=window_size, min_periods=1).mean()

        unique_stocks = set(
            col.split("_buysell")[0]
            for col in evaluation_df.columns
            if "_buysell" in col
        )

        for stock in unique_stocks:
            buysell_col = stock + "_buysell"
            market_col = stock + "_market"

            if (
                buysell_col in evaluation_df.columns
                and market_col in evaluation_df.columns
            ):
                fig, ax1 = plt.subplots(figsize=(14, 7))

                color = "tab:blue"
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Signal", color=color)
                smoothed_signal = moving_average(
                    evaluation_df[buysell_col], window_size=5
                )
                ax1.plot(
                    evaluation_df.index,
                    smoothed_signal,
                    label="Smoothed Signal",
                    color=color,
                    alpha=0.7,
                )
                ax1.tick_params(axis="y", labelcolor=color)

                ax2 = ax1.twinx()
                color = "tab:red"
                ax2.set_ylabel("Market Return", color=color)
                smoothed_market_return = moving_average(
                    evaluation_df[market_col], window_size=5
                )
                ax2.plot(
                    evaluation_df.index,
                    smoothed_market_return,
                    label="Smoothed Market Return",
                    color=color,
                    alpha=0.7,
                )
                ax2.tick_params(axis="y", labelcolor=color)

                fig.tight_layout()
                plt.title(f"Smoothed Signal vs Market Return for {stock}")
                plt.savefig(
                    f"{save_path}/correlation_curves/{stock}_smoothed_signal_vs_market_return.png"
                )
                plt.close()

    def launch(self):
        save_path = './../../data/results/daily_model/'

        self._correlation_by_company()
        self.short_or_long()
        self.evaluate_model_accuracy()
        self.compute_signal_market_correlation()
        self.save_results_to_excel(save_path=save_path)
        self.visualize_courbe(save_path=save_path)


if __name__ == "__main__":
    WEBSCRAPPED_DATA_PATH = (
        "./../../data/new_webscrapping_predicted/concatenated_prediction.csv"
    )
    DAILY_STOCKS_RETURNS_PATH = "./../../data/stocks_daily_data.xlsx"
    analysed_tweets = pd.read_csv(WEBSCRAPPED_DATA_PATH)
    df_returns = pd.read_excel(DAILY_STOCKS_RETURNS_PATH, index_col=0)

    preprocessor = Preprocessing()
    grouped_analysed_tweets, df_returns = preprocessor.process(
        analysed_tweets, df_returns
    )

    model_evaluator = DailyModelEvaluation(grouped_analysed_tweets, df_returns)
    model_evaluator.launch()
