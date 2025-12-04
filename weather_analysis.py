"""
gradebook.py
Simple GradeBook Analyzer CLI
Name: Vansh Negi
Course: B.Tech CSE AI & ML - 1st Year
Section: A
Enrollment Number: 2501730158
Subject: Programming for Problem Solving using Python
# I made this for Lab Assignment 4
"""


# weather_analysis.py
# Robust student-friendly version: auto-detects date column and shows helpful debug info

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

input_file = "weather_sample.csv"
cleaned_file = "weather_sample_cleaned.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def detect_date_column(df):
    # common names to look for
    candidates = ["Date", "date", "Datetime", "datetime", "Day", "day", "DATE"]
    for c in candidates:
        if c in df.columns:
            return c
    # if none match, try to find any column with datetime-like values by trying to parse
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors='coerce')
            # if more than half values parsed as datetimes, assume this is date column
            if parsed.notna().sum() >= max(1, len(parsed) // 2):
                return c
        except Exception:
            continue
    # fallback: use first column
    return df.columns[0]

def load_and_clean():
    print("Loading:", input_file)
    # try reading CSV
    try:
        df = pd.read_csv(input_file)
    except pd.errors.EmptyDataError:
        print("Error: CSV is empty.")
        raise
    except Exception as e:
        # attempt read without header 
        print("Warning: normal read failed, trying without header. Error:", e)
        df = pd.read_csv(input_file, header=None)
        # give temporary column names
        df.columns = [f"col{i}" for i in range(len(df.columns))]
    
    print("\nColumns found in CSV:")
    print(list(df.columns))

    # show first few rows for debugging
    print("\nFirst 5 rows (for debugging):")
    print(df.head(5))

    date_col = detect_date_column(df)
    print(f"\nDetected date column: '{date_col}' (will try to parse it)")

    # parse date
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if df[date_col].isna().all():
        print("Warning: could not parse any dates from detected column. Check your CSV format.")
    # rename parsed date column to 'Date' for consistency
    df = df.rename(columns={date_col: "Date"})

    # drop rows where date couldn't be parsed
    before = len(df)
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f"Dropped {before-after} rows because Date couldn't be parsed.")

    # ensure numeric columns exist, create if missing
    for col in ['Temperature', 'Rainfall', 'Humidity']:
        if col not in df.columns:
            print(f"Note: column '{col}' not found — creating it with NaN values.")
            df[col] = np.nan

    # convert numeric
    for col in ['Temperature', 'Rainfall', 'Humidity']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # fill missing values 
    df['Temperature'] = df['Temperature'].fillna(method='ffill').fillna(method='bfill')
    df['Humidity'] = df['Humidity'].fillna(method='ffill').fillna(method='bfill')
    df['Rainfall'] = df['Rainfall'].fillna(0)

    # sort by date
    df = df.sort_values('Date').reset_index(drop=True)

    return df

def show_stats(df):
    print("\n--- Basic statistics ---")
    print(df[['Temperature', 'Rainfall', 'Humidity']].describe())

def monthly_data(df):
    df['Month'] = df['Date'].dt.to_period('M')
    monthly = df.groupby('Month').agg({
        'Temperature': ['mean', 'min', 'max'],
        'Rainfall': 'sum',
        'Humidity': 'mean'
    })
    # flatten columns
    monthly.columns = ['_'.join(col).strip() for col in monthly.columns.values]
    monthly = monthly.reset_index()
    return monthly

def save_cleaned(df):
    path = os.path.join(OUT_DIR, cleaned_file)
    df.to_csv(path, index=False)
    print("Saved cleaned CSV to", path)

def plot_daily_temperature(df):
    plt.figure(figsize=(10,4))
    plt.plot(df['Date'], df['Temperature'], marker='o', linewidth=1)
    plt.title('Daily Temperature')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "daily_temperature.png")
    plt.savefig(p); plt.close()
    print("Saved", p)

def plot_monthly_rainfall(monthly_df):
    plt.figure(figsize=(8,4))
    x = monthly_df['Month'].astype(str)
    y = monthly_df['Rainfall_sum']
    plt.bar(x, y)
    plt.title('Monthly Rainfall (sum)')
    plt.xlabel('Month')
    plt.ylabel('Rainfall (mm)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "monthly_rainfall.png")
    plt.savefig(p); plt.close()
    print("Saved", p)

def plot_humidity_vs_temperature(df):
    plt.figure(figsize=(6,6))
    plt.scatter(df['Temperature'], df['Humidity'], alpha=0.7)
    plt.title('Humidity vs Temperature')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Humidity (%)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "humidity_vs_temperature.png")
    plt.savefig(p); plt.close()
    print("Saved", p)

def plot_combined(df):
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(df['Date'], df['Temperature'], label='Daily Temp', linewidth=1)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature (°C)')
    ax2 = ax1.twinx()
    ax2.bar(df['Date'], df['Rainfall'], alpha=0.2, label='Daily Rainfall')
    ax2.set_ylabel('Rainfall (mm)')
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "combined_temp_rain.png")
    plt.savefig(p); plt.close()
    print("Saved", p)

def main():
    if not os.path.exists(input_file):
        print("CSV file not found. Place", input_file, "in the same folder as this script.")
        return

    df = load_and_clean()
    show_stats(df)

    save_cleaned(df)
    monthly = monthly_data(df)
    monthly_path = os.path.join(OUT_DIR, "monthly_aggregates.csv")
    monthly.to_csv(monthly_path, index=False)
    print("Saved monthly aggregates to", monthly_path)

    # plots
    plot_daily_temperature(df)
    plot_monthly_rainfall(monthly)
    plot_humidity_vs_temperature(df)
    plot_combined(df)

    print("\nAll done — check the 'outputs' folder.")

if __name__ == "__main__":
    main()
