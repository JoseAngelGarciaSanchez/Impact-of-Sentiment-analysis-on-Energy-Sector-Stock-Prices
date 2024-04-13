import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from parameters import company_to_stock_dict

np.random.seed(42)
pd.options.mode.chained_assignment = None


class MonthlyModelEvaluation:
    
    
#----------- Pre-processing
    def __init__(self, path: str, returns_path: str) -> None:
        self.path = path
        self.returns_path = returns_path

    def read_data(self):
        self.returns = pd.read_excel(
            self.returns_path, sheet_name="Returns", header=[5, 6]
        ).T.iloc[2:, :]
        self.df = pd.read_csv(self.path)

    def format_dates(self):
        """
        - Trim, convert to datetime, and handle missing values for PostDate
        - Extract year and month
        """
        self.df["PostDate"] = pd.to_datetime(self.df["PostDate"].astype(str).str[:-3])
        self.df.dropna(subset=["PostDate"], inplace=True)

        self.df["year"] = self.df["PostDate"].dt.year
        self.df["month"] = self.df["PostDate"].dt.month

    def map_sentiments(self):
        """
        Map sentiment labels to numerical values :

        - positive or bullish -> 1,
        - negative or bearish -> -1,
        - neutral -> 0
        """
        sentiment_mapping = {"Bullish": 1, "Bearish": -1}
        base_sentiment_mapping = {"positive": 1, "neutral": 0, "negative": -1}
        self.df["sentiment"] = self.df["sentiment"].map(sentiment_mapping)
        self.df["sentiment_base"] = self.df["sentiment_base"].map(
            base_sentiment_mapping
        )

    def standardize_column_names(self):
        """Standardize the first row as column names"""
        self.returns.rename(columns=self.returns.iloc[0], inplace=True)
        self.returns = self.returns.iloc[2:]

    def uppercase_column_names(self):
        """Convert all column names to uppercase."""
        self.returns.columns = self.returns.columns.str.upper()

    def reset_and_clean_indices(self):
        """Reset the DataFrame index and rename new indices to meaningful names, remove unnecessary columns."""
        self.returns.reset_index(inplace=True)
        self.returns.rename(columns={"level_1": "DATE"}, inplace=True)
        self.returns.drop(columns="level_0", inplace=True, errors="ignore")

    def format_date_column(self):
        """Convert the 'DATE' column to datetime format."""
        self.returns["DATE"] = pd.to_datetime(self.returns["DATE"])

    def create_year_and_month_columns(self):
        """Extract year and month from 'DATE' and combine into 'yearmonth'."""
        self.returns["year"] = self.returns["DATE"].dt.year
        self.returns["month"] = self.returns["DATE"].dt.month.astype(str).str.zfill(2)
        self.returns["yearmonth"] = (
            self.returns["year"].astype(str) + "-" + self.returns["month"]
        )

    def convert_percentages(self):
        """Convert percentage columns to a numeric scale by excluding date related columns."""
        except_column = ["DATE", "year", "month", "yearmonth"]
        selected_columns = [
            col for col in self.returns.columns if col not in except_column
        ]
        self.returns[selected_columns] = (
            self.returns[selected_columns].astype(float).apply(lambda x: x / 100 + 1)
        )


#----------- Model

    def positive_ratio(self) -> pd.DataFrame:
        """
        Calculate and return the positive sentiment ratio by year, month, and company.

        This function computes the ratio of positive sentiments from tweets grouped by year, 
        month, and company. It also formats the output to include a 'yearmonth' string.

        Returns:
            pd.DataFrame: Contains columns for year, month, company, 'yearmonth', and 'positive_ratio'.
        """
        
        # Group the data by year, month, and company
        grouped = self.df.groupby(["year", "month", "company"])

        # Count sentiments for both 'sentiment' and 'sentiment_base' columns
        sentiments_count = grouped["sentiment"].value_counts().unstack(fill_value=0)
        sentiments_base_count = grouped["sentiment_base"].value_counts().unstack(fill_value=0)

        # Rename columns for clarity
        sentiments_count.columns = [f"sentiment_{col}" for col in sentiments_count.columns]
        sentiments_base_count.columns = [f"sentiment_base_{col}" for col in sentiments_base_count.columns]

        # Merge counts into a single DataFrame
        all_sentiments = pd.concat([sentiments_count, sentiments_base_count], axis=1)

        # Calculate the positive ratio
        all_sentiments['total_tweets'] = all_sentiments.sum(axis=1)
        all_sentiments['positive_tweets'] = all_sentiments[['sentiment_1', 'sentiment_base_1']].sum(axis=1)
        all_sentiments['positive_ratio'] = all_sentiments['positive_tweets'] / all_sentiments['total_tweets']

        # Reset index and prepare final DataFrame
        result = all_sentiments.reset_index()
        result['yearmonth'] = result['year'].astype(str) + '-' + result['month'].astype(str).apply(lambda x: x.zfill(2))

        # Select necessary columns to return
        return result[['year', 'month', 'company', 'yearmonth', 'positive_ratio']]

    def short_or_long(self):
        """
        Constructs a dataframe that indicates whether to buy or sell stocks for each company based on the shifted positive ratio.
        
        The dataframe is indexed by year and month, with columns for each company. A positive 'buy_or_sell' value suggests buying,
        while a negative value suggests selling.
        """
        # Calculate positive ratios by month for each company
        positive_ratios_by_month = self.positive_ratio()
        
        # Extract unique year-month combinations and companies
        date_index = positive_ratios_by_month["yearmonth"].unique().tolist()
        unique_companies = positive_ratios_by_month["company"].unique().tolist()
        
        # Create a dataframe to store buy or sell signals
        self.shortlongdf = pd.DataFrame(index=date_index)
        
        for company in unique_companies:
            # Filter data for the current company
            mask = positive_ratios_by_month["company"] == company
            stock_ratios = positive_ratios_by_month.loc[mask]
            
            # Calculate the shifted positive ratio for buy or sell decision
            stock_ratios["positive_ratio_shifted"] = stock_ratios["positive_ratio"].shift(1)
            stock_ratios["buy_or_sell"] = (stock_ratios["positive_ratio_shifted"] - 0.5) * 2
            
            # Set year-month as index for the company's data
            stock_ratios.set_index("yearmonth", inplace=True)
            
            # Save the buy or sell signals in the main dataframe
            self.shortlongdf[company] = stock_ratios["buy_or_sell"]
            
    def mapping(self):
        """cumsum for all stocks"""
        self.df_columns_list = [
            "BP PLC",
            "FMC CORP",
            "WEYERHAEUSER CO",
            "ALTAGAS LTD",
            "BHP GROUP",
            "INTERNATIONAL PAPER CO",
            "S&P 500 ENERGY INDEX",
            "STORA ENSO",
            "WILMAR INTERNATIONAL LTD",
            "TOTALENERGIES SE",
        ]

        self.stocklist = [
            "BP/ LN EQUITY",
            "FMC US EQUITY",
            "WY US EQUITY",
            "ALA CT EQUITY",
            "BHP US EQUITY",
            "IP US EQUITY",
            "S5ENRS EQUITY",
            "STERV FH EQUITY",
            "WIL SP EQUITY",
            "TTE FP EQUITY",
        ]

        self.search_dictio = {}
        for i, k in enumerate(self.df_columns_list):
            self.search_dictio[self.stocklist[i]] = k

    def adjust_returns_with_company_names(self):
        # Create a DataFrame to hold adjusted returns, matching the structure of shortlongdf
        adjusted_returns = pd.DataFrame()

        for equity_name, company_name in self.search_dictio.items():
            if equity_name in self.returns.columns:
                adjusted_returns[company_name] = self.returns[equity_name]

            adjusted_returns["DATE"] = self.returns.index

        adjusted_returns["year"] = self.returns["year"]
        adjusted_returns["month"] = self.returns["month"]
        adjusted_returns["yearmonth"] = (
            self.returns["year"].astype(str)
            + "-"
            + self.returns["month"].astype(str).str.zfill(2)
        )

        self.adjusted_returns = adjusted_returns.reset_index(drop=True).set_index(
            "yearmonth"
        )


#----------- Evaluate model

    def evaluate_model_accuracy(self):
        self.adjusted_returns.index = pd.to_datetime(self.adjusted_returns.index)
        self.shortlongdf.index = pd.to_datetime(self.shortlongdf.index)

        evaluation_df = self.shortlongdf.join(
            self.adjusted_returns, how="inner", lsuffix="_buysell", rsuffix="_market"
        )
        accuracy_metrics = {}

        def prediction_matches(signal, market_return):
            if signal > 0 and market_return > 1:
                return True
            elif signal < 0 and market_return < 1:
                return True
            elif -0.1 < signal < 0.1 and 0.9 < market_return < 1.1:
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

        # Create a DataFrame from the accuracy metrics dictionary
        metrics_df = pd.DataFrame.from_dict({
            stock: {'Accuracy (%)': (metrics['correct_predictions'] / metrics['total_signals']) * 100 if metrics['total_signals'] > 0 else 0}
            for stock, metrics in accuracy_metrics.items()
        }, orient='index')

        # Save to Excel
        metrics_df.to_excel("monthlymodel_accuracy.xlsx")

        print("Accuracy metrics saved to model_accuracy.xlsx.")

    def compute_signal_market_correlation(self):
        self.adjusted_returns.index = pd.to_datetime(self.adjusted_returns.index)
        self.shortlongdf.index = pd.to_datetime(self.shortlongdf.index)

        evaluation_df = self.shortlongdf.join(
            self.adjusted_returns, how="inner", lsuffix="_signal", rsuffix="_market"
        )

        correlation_results = {}

        # Iterate over columns to compute correlations
        for column in evaluation_df.columns:
            if "_signal" in column:
                signal_column = column
                market_column = column.replace("_signal", "_market")

                # Check if the corresponding market column exists
                if market_column in evaluation_df.columns:
                    # Clean data: remove rows where either column has NaN or inf values
                    clean_df = (
                        evaluation_df[[signal_column, market_column]]
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                    )

                    if not clean_df.empty:
                        # Compute the correlation and its p-value
                        signal_data = clean_df[signal_column]
                        market_data = clean_df[market_column]
                        if len(signal_data) < 2 or len(market_data) < 2:
                            pass
                        else:
                            corr, p_value = pearsonr(signal_data, market_data)

                        company_name = column.split("_signal")[0]

                        # Format the correlation value and include significance if p-value < 0.05
                        significance = "***" if p_value < 0.05 else ""
                        formatted_correlation = f"{corr:.4f} {significance}"

                        correlation_results[company_name] = formatted_correlation

        # Create a DataFrame from the correlation results dictionary
        correlation_df = pd.DataFrame.from_dict(correlation_results, orient='index', columns=['Correlation'])
        
        # Save to Excel
        correlation_df.to_excel("monthly_signal_market_correlation.xlsx")

        print("Correlation results saved to signal_market_correlation.xlsx.")

    def visualize_courbe(self, SAVE_PATH: str | os.PathLike):

        os.makedirs(f"{SAVE_PATH}/correlation_curves/", exist_ok=True)

        evaluation_df = self.shortlongdf.join(
            self.adjusted_returns, how="inner", lsuffix="_buysell", rsuffix="_market"
        )

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
                smoothed_signal = evaluation_df[buysell_col]
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
                smoothed_market_return = evaluation_df[market_col]
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
                    f"{SAVE_PATH}/correlation_curves/{stock}_smoothed_signal_vs_market_return.png"
                )
                plt.close()

    def launch(self):
        self.read_data()
        self.format_dates()
        self.map_sentiments()
        self.standardize_column_names()
        self.uppercase_column_names()
        self.reset_and_clean_indices()
        self.format_date_column()
        self.create_year_and_month_columns()
        self.convert_percentages()
        
        
        self.short_or_long()
        self.mapping()
        self.adjust_returns_with_company_names()
        
        self.evaluate_model_accuracy()
        self.compute_signal_market_correlation()
        self.visualize_courbe(SAVE_PATH="./../data/")


if __name__ == "__main__":
    WEBSCRAPPED_DATA_PATH = "./../data/data_model/all_data.csv"
    DAILY_STOCKS_RETURNS_PATH = "./../data/stocks_data.xlsx"

    analysed_tweets = pd.read_csv(WEBSCRAPPED_DATA_PATH)
    df_returns = pd.read_excel(DAILY_STOCKS_RETURNS_PATH, index_col=0)

    model_evaluator = MonthlyModelEvaluation(
        WEBSCRAPPED_DATA_PATH, DAILY_STOCKS_RETURNS_PATH
    )
    model_evaluator.launch()
