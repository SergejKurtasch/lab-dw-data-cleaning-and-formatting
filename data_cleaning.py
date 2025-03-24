import pandas as pd
import numpy as np

def load_data(url):
    """Load the CSV file into a DataFrame."""
    df = pd.read_csv(url)
    return df

def clean_column_names(df):
    """Standardize column names: lowercase, replace spaces with underscores, and rename specific columns."""
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    df.rename(columns={"st": "state"}, inplace=True)
    return df

def clean_invalid_values(df):
    """Clean inconsistent or incorrect values."""
    df["gender"] = df["gender"].str.strip().replace({"Femal": "F", "female": "F", "Male": "M"})
    df["state"] = df["state"].str.strip().replace({"Cali": "California", "AZ": "Arizona", "WA": "Washington"})
    df["education"] = df["education"].str.strip().replace("Bachelors", "Bachelor")
    df["customer_lifetime_value"] = df["customer_lifetime_value"].str.strip("%").astype(float)
    df["vehicle_class"] = df["vehicle_class"].str.strip().replace(
        {"Luxury Car": "Luxury", "Luxury SUV": "Luxury", "Sports Car": "Luxury"}
    )
    return df

def format_data_types(df):
    """Ensure correct data types for all columns."""
    df["customer_lifetime_value"] = df["customer_lifetime_value"].astype(float)
    df["number_of_open_complaints"] = (
        df["number_of_open_complaints"]
        .astype(str)
        .apply(lambda x: x.split("/")[1] if "/" in x else x)
    )
    df["number_of_open_complaints"] = df["number_of_open_complaints"].replace("nan", np.nan).astype(float)
    return df

def handle_null_values(df):
    """Fill or drop null values."""
    df["customer_lifetime_value"] = df["customer_lifetime_value"].fillna(df["customer_lifetime_value"].mean())
    df["income"] = df["income"].fillna(df["income"].mean())
    df["monthly_premium_auto"] = df["monthly_premium_auto"].fillna(df["monthly_premium_auto"].mean())
    df["number_of_open_complaints"] = df["number_of_open_complaints"].fillna(df["number_of_open_complaints"].mode()[0])
    df["gender"] = df["gender"].fillna(df["gender"].mode()[0])
    df["education"] = df["education"].fillna(df["education"].mode()[0])
    df["policy_type"] = df["policy_type"].fillna(df["policy_type"].mode()[0])
    df["vehicle_class"] = df["vehicle_class"].fillna(df["vehicle_class"].mode()[0])
    df["total_claim_amount"] = df["total_claim_amount"].fillna(df["total_claim_amount"].mean())
    return df

def convert_to_integer(df):
    """Convert all numeric columns to integer data types."""
    df = df.astype({
        "customer_lifetime_value": "int64",
        "income": "int64",
        "monthly_premium_auto": "int64",
        "total_claim_amount": "int64"
    })
    return df

def remove_duplicates(df):
    """Remove duplicate rows and reset the index."""
    df = df.drop_duplicates().reset_index(drop=True)
    return df
