import pandas as pd
import numpy as np
from scipy.stats import pearsonr
np.random.seed(42)  # Pour la reproductibilité
pd.options.mode.chained_assignment = None


class ModelEvaluation:
    def __init__(self, path: str, returns_path: str) -> None:
        self.path = path
        self.returns_path = returns_path

    def read_data(self):
        self.returns = pd.read_excel(
            self.returns_path, sheet_name="Returns", header=[5, 6]
        ).T.iloc[2:, :]
        self.df = pd.read_csv(self.path)

    def formatting(self) -> None:
        """Formatting the dates and encoding sentiment columns positive or bullish -> 1, negative or bearish -> -1, neutral -> 0"""

        self.df["PostDate"] = self.df["PostDate"].astype(
            str).apply(lambda x: x[:-3])
        self.df["PostDate"] = pd.to_datetime(self.df["PostDate"])

        # drop rows with NaN values in the "PostDate" column
        self.df.dropna(subset=["PostDate"], inplace=True)

        # add a column for the year and month
        self.df["year"] = self.df["PostDate"].dt.year
        self.df["month"] = self.df["PostDate"].dt.month

        # formatting sentiments
        self.df["sentiment"] = self.df["sentiment"].map(
            {"Bullish": 1, "Bearish": -1})
        self.df["sentiment_base"] = self.df["sentiment_base"].map(
            {"positive": 1, "neutral": 0, "negative": -1}
        )

        # formatting returns
        self.returns = self.returns.rename(columns=self.returns.iloc[0])
        self.returns = self.returns.iloc[2:]
        def upercase(x): return str(x).upper()
        self.returns.rename(upercase, axis="columns", inplace=True)
        self.returns.reset_index(inplace=True)
        self.returns.rename(
            columns={"level_0": "DATE1", "level_1": "DATE"}, inplace=True
        )
        self.returns.drop(columns="DATE1", inplace=True)
        self.returns["DATE"] = pd.to_datetime(self.returns["DATE"]).dt.date

        # we apply to convert the percentage on indices
        except_column = "DATE"
        selected_columns = [
            col for col in self.returns.columns if col != except_column]
        result = self.returns[selected_columns].apply(
            lambda x: x / 100 + 1, axis=1)
        self.returns = pd.concat([self.returns[except_column], result], axis=1)
        self.returns["DATE"] = pd.to_datetime(self.returns["DATE"])
        self.returns["year"] = self.returns["DATE"].dt.year
        self.returns["month"] = self.returns["DATE"].dt.month
        self.returns["yearmonth"] = (
            self.returns["year"].astype(str)
            + "-"
            + self.returns["month"].astype(str).str.zfill(2)
        )
        result_on_pct = self.returns[selected_columns].apply(
            lambda x: x / 100, axis=1)
        self.returns_on_pct = pd.concat(
            [self.returns[except_column], result_on_pct], axis=1)
        self.returns_on_pct["DATE"] = pd.to_datetime(self.returns["DATE"])
        self.returns_on_pct["year"] = self.returns_on_pct["DATE"].dt.year
        self.returns_on_pct["month"] = self.returns_on_pct["DATE"].dt.month
        self.returns_on_pct["yearmonth"] = (
            self.returns_on_pct["year"].astype(str)
            + "-"
            + self.returns_on_pct["month"].astype(str).str.zfill(2)
        )

    def positive_ratio(self) -> pd.DataFrame:
        # group the data by year and month
        grouped = self.df.groupby(["year", "month", "company"])

        # count the number of positive and negative tweets for each year and month
        sentiment_positive_tweets_by_month = grouped["sentiment"].apply(
            lambda x: (x == 1).sum()
        )
        sentiment_negative_tweets_by_month = grouped["sentiment"].apply(
            lambda x: (x == -1).sum()
        )
        sentiment_base_positive_tweets_by_month = grouped["sentiment_base"].apply(
            lambda x: (x == 1).sum()
        )
        sentiment_base_neutral_tweets_by_month = grouped["sentiment_base"].apply(
            lambda x: (x == 0).sum()
        )
        sentiment_base_negative_tweets_by_month = grouped["sentiment_base"].apply(
            lambda x: (x == -1).sum()
        )

        # calculate the ratio of positive and negative tweets for each year and month
        positive_ratios_by_month = (
            sentiment_base_positive_tweets_by_month + sentiment_positive_tweets_by_month
        ) / (
            sentiment_positive_tweets_by_month
            + sentiment_negative_tweets_by_month
            + sentiment_base_positive_tweets_by_month
            + sentiment_base_neutral_tweets_by_month
            + sentiment_base_negative_tweets_by_month
        )

        # formatting
        positive_ratios_by_month = positive_ratios_by_month.reset_index()
        positive_ratios_by_month.rename(
            columns={0: "positive_ratio"}, inplace=True)
        positive_ratios_by_month["yearmonth"] = (
            positive_ratios_by_month["year"].astype(str)
            + "-"
            + positive_ratios_by_month["month"].astype(str).str.zfill(2)
        )
        return positive_ratios_by_month

    def short_or_long(self):
        """new dataframe with buy or sell at t"""
        positive_ratios_by_month = self.positive_ratio()
        date_index = positive_ratios_by_month["yearmonth"].unique().tolist()
        unique_companies = positive_ratios_by_month["company"].unique(
        ).tolist()
        self.shortlongdf = pd.DataFrame(index=date_index)
        for company in unique_companies:
            # selection of the company
            mask = positive_ratios_by_month["company"] == company
            stock_ratios = positive_ratios_by_month.loc[mask]
            # selection of the period
            stock_ratios.loc[:, "positive_ratio_shifted"] = stock_ratios[
                "positive_ratio"
            ].shift(1)
            stock_ratios.loc[:, "buy_or_sell"] = (
                stock_ratios["positive_ratio_shifted"] - 0.5
            ) * 2
            # STOCK_RATIONS BETWEEN -1 and 1
            stock_ratios.set_index("yearmonth", inplace=True)

            self.stock_ratios = stock_ratios
            # saving info in a dataframe
            self.shortlongdf[company] = stock_ratios["buy_or_sell"]

    """ 
    Pseudo-code : 
    Calculer le signal d'achat ou de vente :
        Utiliser le sentiment du mois précédent (N-1) pour générer un signal d'achat ou de vente.

    Vérifier la performance réelle du titre :
        Si le modèle propose d'acheter (signal > 0), alors vérifier la performance réelle du titre.
            Pour vérifier la performance, comparer le prix du titre au temps t avec son prix au temps t-1 (le mois précédent).
            Si la différence de prix (prix_t - prix_t-1) est positive, alors la performance est jugée positive et le modèle est considéré comme bon pour cette période.
            Sinon, si la différence de prix est négative, la performance est jugée négative et le modèle n'est pas considéré comme bon pour cette période.

    Évaluer la performance du modèle :
        Répéter ce processus pour chaque période et chaque titre concerné.
        Calculer le pourcentage de fois où le modèle a correctement prédit une performance positive lorsque un signal d'achat était donné.
    """

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

            adjusted_returns['DATE'] = self.returns.index
        adjusted_returns['year'] = self.returns['year']
        adjusted_returns['month'] = self.returns['month']
        adjusted_returns['yearmonth'] = self.returns['year'].astype(
            str) + '-' + self.returns['month'].astype(str).str.zfill(2)

        self.adjusted_returns = adjusted_returns.reset_index(
            drop=True).set_index('yearmonth')

    def evaluate_model_accuracy(self):
        self.adjusted_returns.index = pd.to_datetime(
            self.adjusted_returns.index)
        self.shortlongdf.index = pd.to_datetime(self.shortlongdf.index)

        evaluation_df = self.shortlongdf.join(
            self.adjusted_returns, how='inner', lsuffix='_buysell', rsuffix='_market')
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
                if '_buysell' in column:
                    company_name = column.split('_buysell')[0]
                    market_column = company_name + '_market'

                    if market_column in evaluation_df.columns:
                        if company_name not in accuracy_metrics:
                            accuracy_metrics[company_name] = {
                                'correct_predictions': 0, 'total_signals': 0}

                        accuracy_metrics[company_name]['total_signals'] += 1

                        if prediction_matches(row[column], row[market_column]):
                            accuracy_metrics[company_name]['correct_predictions'] += 1

        for stock, metrics in accuracy_metrics.items():
            stock_accuracy = (metrics['correct_predictions'] / metrics['total_signals']
                              ) * 100 if metrics['total_signals'] > 0 else 0
            print(f'{stock} Accuracy: {stock_accuracy:.2f}%')
            

    def compute_signal_market_correlation(self):
        self.adjusted_returns.index = pd.to_datetime(self.adjusted_returns.index)
        self.shortlongdf.index = pd.to_datetime(self.shortlongdf.index)
        
        evaluation_df = self.shortlongdf.join(self.adjusted_returns, how='inner', lsuffix='_signal', rsuffix='_market')
        
        correlation_results = {}
        
        # Iterate over columns to compute correlations
        for column in evaluation_df.columns:
            if '_signal' in column:
                signal_column = column
                market_column = column.replace('_signal', '_market')
                
                # Check if the corresponding market column exists
                if market_column in evaluation_df.columns:
                    # Clean data: remove rows where either column has NaN or inf values
                    clean_df = evaluation_df[[signal_column, market_column]].replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if not clean_df.empty:
                        # Compute the correlation and its p-value
                        signal_data = clean_df[signal_column]
                        market_data = clean_df[market_column]
                        if len(signal_data) < 2 or len(market_data)<2:
                            pass
                        else:
                            corr, p_value = pearsonr(signal_data, market_data)
                        
                        company_name = column.split('_signal')[0]
                        
                        # Format the correlation value and include significance if p-value < 0.05
                        significance = "***" if p_value < 0.05 else ""
                        formatted_correlation = f"{corr:.4f} {significance}"
                        
                        correlation_results[company_name] = formatted_correlation

        # Print the updated correlation results
        for company, corr_value in correlation_results.items():
            print(f"{company} Signal-Market Correlation: {corr_value}")

    def launch(self):
        self.read_data()
        self.formatting()
        self.short_or_long()
        self.mapping()
        self.adjust_returns_with_company_names()
        self.evaluate_model_accuracy()
        self.compute_signal_market_correlation()


if __name__ == "__main__":
    path = "./../data/data_model/all_data.csv"
    returns_path = "./../data/stocks_data.xlsx"
    model = ModelEvaluation(path=path, returns_path=returns_path).launch()
