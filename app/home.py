import streamlit as st
import pandas as pd
from datetime import datetime
from PIL import Image

# from Home import load_data

# Global Variables
DATE_COLUMN = "DATE"
DATA_PATH = "./../data/stocks_data.xlsx"

# Layout
st.set_page_config(page_title="Home", page_icon=":bar_chart:", layout="wide")
st.title("Does Twitter have influence in stocks?")

st.markdown(
    """
 * Use the menu at left to select data and set plot parameters
 * Your plots will appear below
"""
)
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
stock_list = ["bp", "exxonmobil", "diamondbacks", "totalenergies", "upm", "chevron", "newmont"]
for i, stock in enumerate(stock_list):
    logo_path = f"images/{stock}-logo.png"
    if i == 0:
        c1.image(Image.open(logo_path))
    elif i == 1:
        c2.image(Image.open(logo_path))
    elif i == 2:
        c3.image(Image.open(logo_path))
    elif i == 3:
        c4.image(Image.open(logo_path))
    elif i == 4:
        c5.image(Image.open(logo_path))
    elif i == 5:
        c6.image(Image.open(logo_path))
    elif i == 6:
        c7.image(Image.open(logo_path))

@st.cache
def load_data():
    data = pd.read_excel(DATA_PATH, sheet_name="Returns", header=[5, 6]).T.iloc[2:, :]
    data = data.rename(columns=data.iloc[0])
    data = data.iloc[2:]
    upercase = lambda x: str(x).upper()
    data.rename(upercase, axis="columns", inplace=True)
    data.reset_index(inplace=True)
    data.rename(columns={"level_0": "DATE1", "level_1": "DATE"}, inplace=True)
    data.drop(columns="DATE1", inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN]).dt.date
    return data

# Load data into the dataframe
data = load_data()

st.sidebar.markdown("## Select Data Time and Stock")

stocklist = [
    "BP PLC",
    "STORA ENSO OYJ-R SHS",
    "INTERNATIONAL PAPER CO",
    "UPM-KYMMENE OYJ",
    "NEWMONT CORP",
    "EXXON MOBIL CORP",
    "VALERO ENERGY CORP",
    "NUCOR CORP",
    "BARRICK GOLD CORP",
    "FMC CORP",
    "FREEPORT-MCMORAN INC",
    "TOTALENERGIES SE",
    "CONOCOPHILLIPS",
    "ARCHER-DANIELS-MIDLAND CO",
    "POSCO HOLDINGS INC -SPON ADR",
    "BHP GROUP LTD-SPON ADR",
    "TECK RESOURCES LTD-CLS B",
    "RIO TINTO PLC-SPON ADR",
    "WILMAR INTERNATIONAL LTD",
    "MONDI PLC",
    "ANGLO AMERICAN PLC",
    "CENOVUS ENERGY INC",
    "ALTAGAS LTD",
    "WESTLAKE CORP",
    "GLENCORE PLC",
    "MOSAIC CO/THE",
    "MARATHON PETROLEUM CORP",
    "PHILLIPS 66",
    "WEYERHAEUSER CO",
    "ENERGY TRANSFER LP",
    "VIPER ENERGY PARTNERS LP",
    "SUNOCO LP",
    "WESTROCK CO",
    "PEMBINA PIPELINE CORP",
    "ALCOA CORP",
    "ARCELORMITTAL",
    "NUTRIEN CT LTD",
    "NUTRIEN US LTD",
    "DOW INC",
    "CORTEVA INC",
    "OCCIDENTAL PETROLEUM CORP",
    "ONEOK INC",
    "CHEVRON CORP",
    "PIONEER NATURAL RESOURCES CO",
    "TARGA RESOURCES CORP",
    "SCHLUMBERGER LTD",
    "BAKER HUGHES CO",
    "DEVON ENERGY CORP",
    "HESS CORP",
    "MARATHON OIL CORP",
    "WILLIAMS COS INC",
    "COTERRA ENERGY INC",
    "APA CORP",
    "EOG RESOURCES INC",
    "KINDER MORGAN INC",
    "EQT CORP",
    "HALLIBURTON CO",
    "DIAMONDBACK ENERGY INC",
    "S&P 500 ENERGY INDEX",
]

df_columns_list = [
    "BP/ LN Equity",
    "STERV FH Equity",
    "IP US Equity",
    "UPM FH Equity",
    "NEM US Equity",
    "XOM US Equity",
    "VLO US Equity",
    "NUE US Equity",
    "ABX CT Equity",
    "FMC US Equity",
    "FCX US Equity",
    "TTE FP Equity",
    "COP US Equity",
    "ADM US Equity",
    "PKX US Equity",
    "BHP US Equity",
    "TECK/B CT Equity",
    "RIO US Equity",
    "WIL SP Equity",
    "MNDI LN Equity",
    "AAL LN Equity",
    "CVE CT Equity",
    "ALA CT Equity",
    "WLK US Equity",
    "GLEN LN Equity",
    "MOS US Equity",
    "MPC US Equity",
    "PSX US Equity",
    "WY US Equity",
    "ET US Equity",
    "VNOM UW Equity",
    "SUN US Equity",
    "WRK US Equity",
    "PBA US Equity",
    "AA US Equity",
    "MTS SQ Equity",
    "NTR CT Equity",
    "NTR US Equity",
    "DOW US Equity",
    "CTVA US Equity",
    "OXY US Equity",
    "OKE US Equity",
    "CVX US Equity",
    "PXD US Equity",
    "TRGP US Equity",
    "SLB US Equity",
    "BKR US Equity",
    "DVN US Equity",
    "HES US Equity",
    "MRO US Equity",
    "WMB US Equity",
    "CTRA US Equity",
    "APA US Equity",
    "EOG US Equity",
    "KMI US Equity",
    "EQT US Equity",
    "HAL US Equity",
    "FANG US Equity",
    "S5ENRS Index",
]

search_dictio = {}
for i, k in enumerate(df_columns_list):
    search_dictio[stocklist[i]] = k

# -- Choose stocks from stockslists
stocks_selected = st.sidebar.multiselect("Stock list", stocklist)

stocks = []
stocks.append("DATE")
for i in stocks_selected:
    stocks.append(search_dictio.get(i).upper())

# -- Choose the start and end dates
min_date = data[DATE_COLUMN].min()
max_date = data[DATE_COLUMN].max()
start_date = st.sidebar.date_input(
    "Start date:", min_value=min_date, max_value=max_date, value=min_date
)
end_date = st.sidebar.date_input(
    "End date:", min_value=min_date, max_value=max_date, value=max_date
)
st.write("Selected range:", start_date, "to", end_date)

# -- Show data
# ----- Conditions
condition1 = start_date <= data[DATE_COLUMN]
condition2 = data[DATE_COLUMN] <= end_date
mask = condition1 & condition2
filtered_df = data.loc[mask, :]
# stocks_df = filtered_df[stocks]
# # ----- Reporting
# m1, m2, m3, m4, m5 = st.columns((1,1,1,1,1))

# turnover = stocks_df[start_date]
# ch = stocks_df[(filtered_df['Hospital Attended']==hosp) & (filtered_df['Metric']== 'Current Handover Average Mins')]   
# hl = stocks_df[(filtered_df['Hospital Attended']==hosp) & (filtered_df['Metric']== 'Hours Lost to Handovers Over 15 Mins')]

# m1.write('')
# m2.metric(label ='Total Outstanding Handovers',value = int(to['Value']), delta = str(int(to['Previous']))+' Compared to 1 hour ago', delta_color = 'inverse')
# m3.metric(label ='Current Handover Average',value = str(int(ch['Value']))+" Mins", delta = str(int(ch['Previous']))+' Compared to 1 hour ago', delta_color = 'inverse')
# m4.metric(label = 'Time Lost today (Above 15 mins)',value = str(int(hl['Value']))+" Hours", delta = str(int(hl['Previous']))+' Compared to yesterday')
# m1.write('')
# ---- Plot
st.dataframe(filtered_df[stocks])
st.line_chart(
    data=filtered_df, x="DATE", y=stocks, width=0, height=0, use_container_width=True
)

# -- Aknowledgments
c1, c2 = st.columns(2)
with c1:
    st.info(
        "**Data Scientists Linkedin: [@JAGS](https://www.linkedin.com/in/jagarciasanchez) [@Sarra](https://www.linkedin.com/in/sarrabenyahia/)**",
        icon="💡",
    )
with c2:
    st.info(
        "**GitHub: [@JAGS](https://github.com/Pse1234) [@Sarra](https://github.com/sarrabenyahia)**",
        icon="💻",
    )