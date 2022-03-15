# Made by caglakndmr @GitHub
# A mini project for association rule learning, to better understand the concept.


# Dataset used
#
# Online Retail Dataset
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Attribute Information:
#
# InvoiceNo: Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code
# starts with the letter 'c', it indicates a cancellation.
# StockCode: Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product.
# Description: Product (item) name. Nominal.
# Quantity: The quantities of each product (item) per transaction. Numeric.
# InvoiceDate: Invice date and time. Numeric. The day and time when a transaction was generated.
# UnitPrice: Unit price. Numeric. Product price per unit in sterling (Â£).
# CustomerID: Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer.
# Country: Country name. Nominal. The name of the country where a customer resides.


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.describe().T
df.isnull().sum()
df.shape


def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


df = retail_data_prep(df)


def outlier_thresholds(dataframe, variable):
    quantile1 = dataframe[variable].quantile(0.01)
    quantile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df = retail_data_prep(df)
df.isnull().sum()
df.describe().T



# Association Rule Learning

# only France
df_fr = df[df["Country"] == "France"]
df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).head(20)

# pivot, so we can have products on columns
df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]
df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0).\
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]
df_fr.groupby(["Invoice", "StockCode"]).agg({"Quantity": "sum"}).unstack().fillna(0).\
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(["Invoice", "StockCode"])["Quantity"].sum().unstack().\
                   fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["Invoice", "Description"])["Quantity"].sum().unstack(). \
            fillna(0).applymap(lambda x: 1 if x > 0 else 0)


fr_inv_pro_df = create_invoice_product_df(df_fr)
fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


check_id(df_fr, 10002)



# Rules
frequent_itemsets = apriori(fr_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"] > 5)]

check_id(df_fr, 21086)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"] > 5)].sort_values("confidence", ascending=False)



# Product Recommendation for carts
product_id = 22492
check_id(df, product_id)

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, 22492, 1)
arl_recommender(rules, 22492, 2)
arl_recommender(rules, 22492, 3)
