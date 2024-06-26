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

    from scipy.stats import mode

# Define a custom rolling mode function using scipy.stats.mode
def rolling_mode(series):
    mode_values = mode(series, axis=None)
    return mode_values.mode[0] if mode_values.count[0] > 0 else np.nan

# Apply the custom rolling mode function
data["11_month_rolling_mode_perf"] = data.groupby("emp_id")["performance_values_ff"].transform(
    lambda x: x.rolling(window=11, min_periods=1).apply(rolling_mode)


    
)


    import pandas as pd
import numpy as np
from scipy.stats import mode

class AttritionRiskTransformer(BaseTransformer):
    def perform_custom_calculations(self):
        """
        Converts attrition risk description into a mode of the last 3 months.
        
        Returns:
            pd.DataFrame: A dataframe containing the following columns:
                - emp_id
                - attrn_risk_desc
                - attrn_risk_factor
                - attrn_risk_mode_3_months
        """
        # Extract and fill missing data
        data = (
            self.extractor.mapping.get('AttritionRiskExtractor')
            .extract_data()
            .fillna("Not Available")
        )

        # Convert all columns to lowercase
        data.columns = [x.lower() for x in data.columns]

        # Convert date columns to datetime
        data['mstr_dt'] = pd.to_datetime(data['mstr_dt'])

        # Ensure attrition risk factor is categorical
        data['attrn_risk_factor'] = data['attrn_risk_desc'].astype('category').cat.codes

        # Calculate the mode of attrition risk factor over a 3-month rolling window
        data['attrn_risk_mode_3_months'] = self.rolling_mode(data['attrn_risk_factor'], window=3)

        # Map attrition risk modes to descriptive values
        attrn_dict = {0: "High", 1: "Low", 2: "Medium", 3: "Not Available"}
        data['attrn_risk_mode_3_months'] = data['attrn_risk_mode_3_months'].map(attrn_dict)

        return data

    @staticmethod
    def rolling_mode(series, window):
        """
        Compute the rolling mode of a Pandas Series.
        
        Args:
            series (pd.Series): The series to compute the rolling mode on.
            window (int): The window size for computing the rolling mode.
        
        Returns:
            pd.Series: A series containing the rolling mode.
        """
        # Create a padded array for rolling window
        padded_series = np.pad(series, (window - 1, 0), mode='constant', constant_values=np.nan)
        shape = (series.size, window)
        strides = padded_series.strides[0]

        rolling_matrix = np.lib.stride_tricks.as_strided(padded_series, shape=shape, strides=(strides, strides))
        mode_result, _ = mode(rolling_matrix, axis=1, nan_policy='omit')

        return pd.Series(mode_result.flatten(), index=series.index).fillna(3)  # Fill NaNs with 'Not Available' code

class AttritionTeamTransformer(BaseTransformer):






                import pandas as pd
import numpy as np
from scipy.stats import mode

class AttritionRiskTransformer(BaseTransformer):
    def perform_custom_calculations(self):
        """
        Converts attrition risk description into a mode of the last 3 months.
        
        Returns:
            pd.DataFrame: A dataframe containing the following columns:
                - emp_id
                - mstr_dt
                - attrn_risk_desc
                - attrn_risk_factor
                - attrn_risk_mode_3_months
        """
        # Extract and fill missing data
        data = (
            self.extractor.mapping.get('AttritionRiskExtractor')
            .extract_data()
            .fillna("Not Available")
        )

        # Convert all columns to lowercase
        data.columns = [x.lower() for x in data.columns]

        # Rename columns for consistency
        data = data.rename(columns={"snap_dt": "mstr_dt"})

        # Convert date columns to datetime
        data['mstr_dt'] = pd.to_datetime(data['mstr_dt'])

        # Ensure attrition risk factor is categorical
        data['attrn_risk_factor'] = data['attrn_risk_desc'].astype('category').cat.codes

        # Calculate the mode of attrition risk factor over a 3-month rolling window
        data['attrn_risk_mode_3_months'] = self.rolling_mode(data['attrn_risk_factor'].values, window=3)

        # Map attrition risk modes to descriptive values
        attrn_dict = {0: "High", 1: "Low", 2: "Medium", 3: "Not Available"}
        data['attrn_risk_mode_3_months'] = data['attrn_risk_mode_3_months'].map(attrn_dict)

        return data

    @staticmethod
    def rolling_mode(arr, window):
        """
        Compute the rolling mode of a numpy array.
        
        Args:
            arr (np.ndarray): The array to compute the rolling mode on.
            window (int): The window size for computing the rolling mode.
        
        Returns:
            np.ndarray: An array containing the rolling mode.
        """
        if len(arr) < window:
            return np.full(len(arr), 3)  # If there are fewer elements than the window size, return 'Not Available' code

        result = np.full(len(arr), 3)  # Default to 'Not Available' code
        for i in range(window - 1, len(arr)):
            window_slice = arr[i - window + 1:i + 1]
            most_common = mode(window_slice).mode[0]
            result[i] = most_common

        return result

class AttritionTeamTransformer(BaseTransformer):
    pass
    pass
