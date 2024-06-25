import dask.dataframe as dd
import numpy as np
import pandas as pd

# Assuming you have a pandas DataFrame `data` loaded from `self.extractor_mapping.get("PerfHistExtractor").extract_data()`
data = self.extractor_mapping.get("PerfHistExtractor").extract_data()

# Convert to Dask DataFrame
data = dd.from_pandas(data, npartitions=10)

# Cast all columns to lowercase
data.columns = [x.lower() for x in data.columns]

# Rename and type conversion
data = data.rename(columns={"snap_dt": "mstr_dt"})
data["mstr_dt"] = dd.to_datetime(data["mstr_dt"])
data["year"] = data["mstr_dt"].dt.year
data["month"] = data["mstr_dt"].dt.month
data["day"] = data["mstr_dt"].dt.day

data["performance"] = data["performance"].fillna(0)
data["promo_ind"] = data["promo_ind"].fillna(0)
data = data.drop_duplicates(subset=["mstr_dt", "emp_id"]).reset_index(drop=True)

# Map performance to integers
perf_dict = {"Below Strong": 1, "Strong": 2, "Above Strong": 3}
data["performance_values"] = data["performance"].map(perf_dict)

# Sort data
data = data.sort_values(["emp_id", "mstr_dt"])

# Calculate performance value differences
data["perf_values_diff"] = data.groupby("emp_id")["performance_values"].apply(lambda x: x.diff(), meta=('x', 'f8'))

# Calculate mom_performance_change
data["mom_performance_change"] = data["perf_values_diff"].map_partitions(
    lambda df: np.where(df.isnull(), np.nan, np.where(df == 0, 0, np.where(df > 0, 1, -1))), meta=('x', 'f8')
)

# Calculate increase and decrease indicators
data["mom_performance_change_inc"] = data["mom_performance_change"].map_partitions(lambda df: np.where(df > 0, 1, 0), meta=('x', 'f8'))
data["mom_performance_change_dec"] = data["mom_performance_change"].map_partitions(lambda df: np.where(df < 0, 1, 0), meta=('x', 'f8'))

# Set the index to mstr_dt for time-based rolling operations and repartition
data = data.set_index('mstr_dt')
data = data.repartition(freq='M')

# Rolling calculations (with time-based index)
data["perf_increase_count_18_months"] = data.groupby("emp_id")["mom_performance_change_inc"].rolling('548D').sum().reset_index(drop=True)
data["perf_decrease_count_18_months"] = data.groupby("emp_id")["mom_performance_change_dec"].rolling('548D').sum().reset_index(drop=True)

# Reset index to default
data = data.reset_index()

# Cumulative sums for performance categories
data["performance_category_strong_int"] = (data["performance"] == "Strong").astype(int)
data["performance_category_strong_cumsum"] = data.groupby("emp_id")["performance_category_strong_int"].cumsum()

data["performance_category_below_strong_int"] = (data["performance"] == "Below Strong").astype(int)
data["performance_category_below_strong_cumsum"] = data.groupby("emp_id")["performance_category_below_strong_int"].cumsum()

data["performance_category_above_strong_int"] = (data["performance"] == "Above Strong").astype(int)
data["performance_category_above_strong_cumsum"] = data.groupby("emp_id")["performance_category_above_strong_int"].cumsum()

data["fraction_of_above_strong_from_start"] = data.map_partitions(
    lambda df: df["performance_category_above_strong_cumsum"].astype(float) /
    (df["performance_category_below_strong_cumsum"].astype(float) +
     df["performance_category_above_strong_cumsum"].astype(float) +
     df["performance_category_strong_cumsum"].astype(float)),
    meta=('x', 'f8')
)

data["fraction_of_below_strong_from_start"] = data.map_partitions(
    lambda df: df["performance_category_below_strong_cumsum"].astype(float) /
    (df["performance_category_below_strong_cumsum"].astype(float) +
     df["performance_category_above_strong_cumsum"].astype(float) +
     df["performance_category_strong_cumsum"].astype(float)),
    meta=('x', 'f8')
)

data["performance_values_ff"] = data["performance_values"].ffill()

data["11_month_rolling_mode_perf"] = data.groupby("emp_id")["performance_values_ff"].rolling('11M').apply(lambda x: pd.Series.mode(x)[0], raw=True).reset_index(drop=True)

# Compute all at once to optimize
data = data.compute()

    -----------------



import dask.dataframe as dd
import numpy as np

# Assuming you have a pandas DataFrame `data` loaded from `self.extractor_mapping.get("PerfHistExtractor").extract_data()`
data = self.extractor_mapping.get("PerfHistExtractor").extract_data()

# Convert to Dask DataFrame
data = dd.from_pandas(data, npartitions=10)

# Cast all columns to lowercase
data.columns = [x.lower() for x in data.columns]

# Rename and type conversion
data = data.rename(columns={"snap_dt": "mstr_dt"})
data["mstr_dt"] = dd.to_datetime(data["mstr_dt"])
data["year"] = data["mstr_dt"].dt.year
data["month"] = data["mstr_dt"].dt.month
data["day"] = data["mstr_dt"].dt.day

data["performance"] = data["performance"].fillna(0)
data["promo_ind"] = data["promo_ind"].fillna(0)
data = data.drop_duplicates(subset=["mstr_dt", "emp_id"]).reset_index(drop=True)

# Map performance to integers
perf_dict = {"Below Strong": 1, "Strong": 2, "Above Strong": 3}
data["performance_values"] = data["performance"].map(perf_dict)

# Calculate performance value differences
data = data.sort_values(["emp_id", "mstr_dt"])
data["perf_values_diff"] = data.groupby("emp_id")["performance_values"].apply(lambda x: x.diff(), meta=('x', 'f8'))

# Calculate mom_performance_change
data["mom_performance_change"] = data["perf_values_diff"].map_partitions(
    lambda df: np.where(df.isnull(), np.nan, np.where(df == 0, 0, np.where(df > 0, 1, -1))), meta=('x', 'f8')
)

# Calculate increase and decrease indicators
data["mom_performance_change_inc"] = data["mom_performance_change"].map_partitions(lambda df: np.where(df > 0, 1, 0), meta=('x', 'f8'))
data["mom_performance_change_dec"] = data["mom_performance_change"].map_partitions(lambda df: np.where(df < 0, 1, 0), meta=('x', 'f8'))

# Rolling calculations
data["perf_increase_count_18_months"] = data.groupby("emp_id")["mom_performance_change_inc"].rolling(window=18).sum().reset_index(0, drop=True)
data["perf_decrease_count_18_months"] = data.groupby("emp_id")["mom_performance_change_dec"].rolling(window=18).sum().reset_index(0, drop=True)

# Cumulative sums for performance categories
data["performance_category_strong_int"] = (data["performance"] == "Strong").astype(int)
data["performance_category_strong_cumsum"] = data.groupby("emp_id")["performance_category_strong_int"].cumsum()

data["performance_category_below_strong_int"] = (data["performance"] == "Below Strong").astype(int)
data["performance_category_below_strong_cumsum"] = data.groupby("emp_id")["performance_category_below_strong_int"].cumsum()

data["performance_category_above_strong_int"] = (data["performance"] == "Above Strong").astype(int)
data["performance_category_above_strong_cumsum"] = data.groupby("emp_id")["performance_category_above_strong_int"].cumsum()

data["fraction_of_above_strong_from_start"] = data.map_partitions(
    lambda df: df["performance_category_above_strong_cumsum"].astype(float) /
    (df["performance_category_below_strong_cumsum"].astype(float) +
     df["performance_category_above_strong_cumsum"].astype(float) +
     df["performance_category_strong_cumsum"].astype(float)),
    meta=('x', 'f8')
)

data["fraction_of_below_strong_from_start"] = data.map_partitions(
    lambda df: df["performance_category_below_strong_cumsum"].astype(float) /
    (df["performance_category_below_strong_cumsum"].astype(float) +
     df["performance_category_above_strong_cumsum"].astype(float) +
     df["performance_category_strong_cumsum"].astype(float)),
    meta=('x', 'f8')
)

data["performance_values_ff"] = data["performance_values"].ffill()

data["11_month_rolling_mode_perf"] = data.groupby("emp_id")["performance_values_ff"].rolling(window=11).apply(lambda x: pd.Series.mode(x)[0], raw=True).reset_index(0, drop=True)

# Compute all at once to optimize
data = data.compute()


    --------------------------------------


    import dask.dataframe as dd
import numpy as np
import pandas as pd

# Assuming you have a pandas DataFrame `data` loaded from `self.extractor_mapping.get("PerfHistExtractor").extract_data()`
data = self.extractor_mapping.get("PerfHistExtractor").extract_data()

# Convert to Dask DataFrame
data = dd.from_pandas(data, npartitions=10)

# Cast all columns to lowercase
data.columns = [x.lower() for x in data.columns]

# Rename and type conversion
data = data.rename(columns={"snap_dt": "mstr_dt"})
data["mstr_dt"] = dd.to_datetime(data["mstr_dt"])
data["year"] = data["mstr_dt"].dt.year
data["month"] = data["mstr_dt"].dt.month
data["day"] = data["mstr_dt"].dt.day

data["performance"] = data["performance"].fillna(0)
data["promo_ind"] = data["promo_ind"].fillna(0)
data = data.drop_duplicates(subset=["mstr_dt", "emp_id"]).reset_index(drop=True)

# Map performance to integers
perf_dict = {"Below Strong": 1, "Strong": 2, "Above Strong": 3}
data["performance_values"] = data["performance"].map(perf_dict)

# Sort data
data = data.sort_values(["emp_id", "mstr_dt"])

# Calculate performance value differences
data["perf_values_diff"] = data.groupby("emp_id")["performance_values"].apply(lambda x: x.diff(), meta=('x', 'f8'))

# Calculate mom_performance_change
data["mom_performance_change"] = data["perf_values_diff"].map_partitions(
    lambda df: np.where(df.isnull(), np.nan, np.where(df == 0, 0, np.where(df > 0, 1, -1))), meta=('x', 'f8')
)

# Calculate increase and decrease indicators
data["mom_performance_change_inc"] = data["mom_performance_change"].map_partitions(lambda df: np.where(df > 0, 1, 0), meta=('x', 'f8'))
data["mom_performance_change_dec"] = data["mom_performance_change"].map_partitions(lambda df: np.where(df < 0, 1, 0), meta=('x', 'f8'))

# Set the index to mstr_dt for time-based rolling operations
data = data.set_index('mstr_dt')

# Rolling calculations (with time-based index)
data["perf_increase_count_18_months"] = data.groupby("emp_id")["mom_performance_change_inc"].rolling('18M').sum().reset_index(0, drop=True)
data["perf_decrease_count_18_months"] = data.groupby("emp_id")["mom_performance_change_dec"].rolling('18M').sum().reset_index(0, drop=True)

# Reset index to default
data = data.reset_index()

# Cumulative sums for performance categories
data["performance_category_strong_int"] = (data["performance"] == "Strong").astype(int)
data["performance_category_strong_cumsum"] = data.groupby("emp_id")["performance_category_strong_int"].cumsum()

data["performance_category_below_strong_int"] = (data["performance"] == "Below Strong").astype(int)
data["performance_category_below_strong_cumsum"] = data.groupby("emp_id")["performance_category_below_strong_int"].cumsum()

data["performance_category_above_strong_int"] = (data["performance"] == "Above Strong").astype(int)
data["performance_category_above_strong_cumsum"] = data.groupby("emp_id")["performance_category_above_strong_int"].cumsum()

data["fraction_of_above_strong_from_start"] = data.map_partitions(
    lambda df: df["performance_category_above_strong_cumsum"].astype(float) /
    (df["performance_category_below_strong_cumsum"].astype(float) +
     df["performance_category_above_strong_cumsum"].astype(float) +
     df["performance_category_strong_cumsum"].astype(float)),
    meta=('x', 'f8')
)

data["fraction_of_below_strong_from_start"] = data.map_partitions(
    lambda df: df["performance_category_below_strong_cumsum"].astype(float) /
    (df["performance_category_below_strong_cumsum"].astype(float) +
     df["performance_category_above_strong_cumsum"].astype(float) +
     df["performance_category_strong_cumsum"].astype(float)),
    meta=('x', 'f8')
)

data["performance_values_ff"] = data["performance_values"].ffill()

data["11_month_rolling_mode_perf"] = data.groupby("emp_id")["performance_values_ff"].rolling('11M').apply(lambda x: pd.Series.mode(x)[0], raw=True).reset_index(0, drop=True)

# Compute all at once to optimize
data = data.compute()
---------------------------------------


    import dask.dataframe as dd
import numpy as np
import pandas as pd

# Assuming you have a pandas DataFrame `data` loaded from `self.extractor_mapping.get("PerfHistExtractor").extract_data()`
data = self.extractor_mapping.get("PerfHistExtractor").extract_data()

# Convert to Dask DataFrame
data = dd.from_pandas(data, npartitions=10)

# Cast all columns to lowercase
data.columns = [x.lower() for x in data.columns]

# Rename and type conversion
data = data.rename(columns={"snap_dt": "mstr_dt"})
data["mstr_dt"] = dd.to_datetime(data["mstr_dt"])
data["year"] = data["mstr_dt"].dt.year
data["month"] = data["mstr_dt"].dt.month
data["day"] = data["mstr_dt"].dt.day

data["performance"] = data["performance"].fillna(0)
data["promo_ind"] = data["promo_ind"].fillna(0)
data = data.drop_duplicates(subset=["mstr_dt", "emp_id"]).reset_index(drop=True)

# Map performance to integers
perf_dict = {"Below Strong": 1, "Strong": 2, "Above Strong": 3}
data["performance_values"] = data["performance"].map(perf_dict)

# Sort data
data = data.sort_values(["emp_id", "mstr_dt"])

# Calculate performance value differences
data["perf_values_diff"] = data.groupby("emp_id")["performance_values"].apply(lambda x: x.diff(), meta=('x', 'f8'))

# Calculate mom_performance_change
data["mom_performance_change"] = data["perf_values_diff"].map_partitions(
    lambda df: np.where(df.isnull(), np.nan, np.where(df == 0, 0, np.where(df > 0, 1, -1))), meta=('x', 'f8')
)

# Calculate increase and decrease indicators
data["mom_performance_change_inc"] = data["mom_performance_change"].map_partitions(lambda df: np.where(df > 0, 1, 0), meta=('x', 'f8'))
data["mom_performance_change_dec"] = data["mom_performance_change"].map_partitions(lambda df: np.where(df < 0, 1, 0), meta=('x', 'f8'))

# Set the index to mstr_dt for time-based rolling operations
data = data.set_index('mstr_dt')

# Rolling calculations (with time-based index)
data["perf_increase_count_18_months"] = data.groupby("emp_id")["mom_performance_change_inc"].rolling('18M').sum().reset_index(0, drop=True)
data["perf_decrease_count_18_months"] = data.groupby("emp_id")["mom_performance_change_dec"].rolling('18M').sum().reset_index(0, drop=True)

# Reset index to default
data = data.reset_index()

# Cumulative sums for performance categories
data["performance_category_strong_int"] = (data["performance"] == "Strong").astype(int)
data["performance_category_strong_cumsum"] = data.groupby("emp_id")["performance_category_strong_int"].cumsum()

data["performance_category_below_strong_int"] = (data["performance"] == "Below Strong").astype(int)
data["performance_category_below_strong_cumsum"] = data.groupby("emp_id")["performance_category_below_strong_int"].cumsum()

data["performance_category_above_strong_int"] = (data["performance"] == "Above Strong").astype(int)
data["performance_category_above_strong_cumsum"] = data.groupby("emp_id")["performance_category_above_strong_int"].cumsum()

data["fraction_of_above_strong_from_start"] = data.map_partitions(
    lambda df: df["performance_category_above_strong_cumsum"].astype(float) /
    (df["performance_category_below_strong_cumsum"].astype(float) +
     df["performance_category_above_strong_cumsum"].astype(float) +
     df["performance_category_strong_cumsum"].astype(float)),
    meta=('x', 'f8')
)

data["fraction_of_below_strong_from_start"] = data.map_partitions(
    lambda df: df["performance_category_below_strong_cumsum"].astype(float) /
    (df["performance_category_below_strong_cumsum"].astype(float) +
     df["performance_category_above_strong_cumsum"].astype(float) +
     df["performance_category_strong_cumsum"].astype(float)),
    meta=('x', 'f8')
)

data["performance_values_ff"] = data["performance_values"].ffill()

data["11_month_rolling_mode_perf"] = data.groupby("emp_id")["performance_values_ff"].rolling('11M').apply(lambda x: pd.Series.mode(x)[0], raw=True).reset_index(0, drop=True)

# Compute all at once to optimize
data = data.compute()
def rolling_mode(arr, window):
    result = np.empty(len(arr))
    result[:] = np.nan
    for i in range(window, len(arr) + 1):
        result[i - 1] = pd.Series(arr[i - window:i]).mode()[0]
    return result

data['18_month_rolling_mode_perf'] = rolling_mode(data['performance_values_ff'].values, 18)
