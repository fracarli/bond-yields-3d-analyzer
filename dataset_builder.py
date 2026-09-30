from datetime import datetime
import numpy as np
import pandas as pd

# =========================================================================
MIN_MATURITY = 1  # min (ex. 1 year)
MAX_MATURITY = 10  # max (ex. 10 years)

START_DATE = '1990-01-01'  # initial historical date for FRED data
END_DATE = '2026-09-30'  # final historical date for FRED data
# =========================================================================


def download_and_build_ml_dataset(min_mat, max_mat, start_date, end_date):
  """Downloads official Zero-Coupon yield curves from FRED based on customized

  maturity ranges and dates, computes initial prices (P0) and future prices (P1)
  using a 1-year time-shift logic, and exports a clean dataset ready for ML.
  """
  print(
      'Connecting to FRED to download Zero-Coupon yield curves '
      f'({min_mat}-{max_mat}Y)...'
  )

  # 1. Define official FRED series URLs dynamically based on configuration
  urls = {
      i: f'https://fred.stlouisfed.org/graph/fredgraph.csv?id=THREEFY{i}'
      for i in range(min_mat, max_mat + 1)
  }

  # 2. Fetch and merge all curves into a single synchronized DataFrame
  df_all = pd.DataFrame()
  for maturity, url in urls.items():
    df_temp = pd.read_csv(url)
    val_col = df_temp.columns[1]

    # Clean non-numeric values (FRED sometimes uses '.' for missing data)
    df_temp[val_col] = pd.to_numeric(
        df_temp[val_col].astype(str).str.strip(), errors='coerce'
    )

    df_temp.columns = ['DATE', f'Rate_{maturity}Y']
    df_temp['DATE'] = pd.to_datetime(df_temp['DATE'])
    df_temp.set_index('DATE', inplace=True)

    if df_all.empty:
      df_all = df_temp
    else:
      df_all = df_all.join(df_temp, how='inner')

  # Drop missing rows after merging all maturities
  df_all.dropna(inplace=True)

  # Filter date range
  df_all = df_all.loc[start_date:end_date]
  print(
      'Historical yield curves loaded for range '
      f'{start_date} to {end_date}: {len(df_all)} trading days.'
  )


  records = []
  trading_days_shift = 252  # Approximate number of trading days in 1 year

  # Determine allowed initial durations T.
  # T must start at least from min_mat + 1 (since we need T-1 available in our curves)
  # and cannot exceed max_mat.
  effective_min_T = max(min_mat + 1, 2)
  effective_max_T = max_mat

  # Iterate through the history leaving enough space for the 1-year future shift
  for i in range(len(df_all) - trading_days_shift):
    row_t0 = df_all.iloc[i]
    row_t1 = df_all.iloc[i + trading_days_shift]
    current_date = df_all.index[i]

    for T in range(effective_min_T, effective_max_T + 1):
      # --- Feature Extraction at time t0 ---
      r_t0 = row_t0[f'Rate_{T}Y']
      # Compute Initial Price P0 using Zero-Coupon pricing formula: P = 100 / (1 + r)^T
      p_0 = 100.0 / ((1.0 + r_t0 / 100.0) ** T)

      # --- Target Extraction at time t1 (1 year later) ---
      # One year later, the same bond has a remaining duration of (T - 1)
      r_t1 = row_t1[f'Rate_{T-1}Y']
      p_1 = 100.0 / ((1.0 + r_t1 / 100.0) ** (T - 1))

      # Append structured record matching requested output columns
      records.append({
          'date': current_date.strftime('%Y-%m-%d'),
          'duration': T,
          'initial_price_p0': round(p_0, 2),
          'interest_rate': round(r_t0, 4),
          'target_price_p1': round(p_1, 2),
      })

  # 3. Create final DataFrame and export to CSV
  df_dataset = pd.DataFrame(records)
  output_filename = 'ml_training_dataset.csv'
  df_dataset.to_csv(output_filename, index=False)

  print(f"Dataset successfully exported to '{output_filename}'!")
  return df_dataset


if __name__ == '__main__':
  dataset_ml = download_and_build_ml_dataset(
      MIN_MATURITY, MAX_MATURITY, START_DATE, END_DATE
  )

  print(f'Total training samples ready: {len(dataset_ml)}')