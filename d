
import pandas as pd
import numpy as np

# Parameters
num_employees = 10  # Number of employees
months = pd.date_range(start='2024-07-31', periods=6, freq='M')  # Next 6 end-of-month dates

# Generate employee IDs
employee_ids = [f'EMP{str(i).zfill(4)}' for i in range(1, num_employees + 1)]

# Function to generate probabilities
def generate_probabilities(num_months):
    probs = np.random.rand(num_months)
    return probs / probs.sum()

# Generate data
data = []
for emp_id in employee_ids:
    probs = generate_probabilities(len(months))
    for month, prob in zip(months, probs):
        data.append({'Employee ID': emp_id, 'Date': month, 'Attrition Probability': prob})

# Create DataFrame
df = pd.DataFrame(data)

# Display DataFrame
print(df)

import pandas as pd

# Sample DataFrames with mixed data types
df1 = pd.DataFrame({
    'employee_id': [1, 2, 3, 4],
    'start_date': ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01'],
    'end_date': ['2022-01-31', '2022-02-28', '2022-03-31', '2022-04-30'],
    'feature1': ['10', '20', '30', 'forty'],  # Mix of numeric and string values
    'feature2': ['100', '200', 'three hundred', '400']  # Mix of numeric and string values
})

df2 = pd.DataFrame({
    'employee_id': [1, 2, 3, 4],
    'start_date': ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01'],
    'end_date': ['2022-01-31', '2022-02-28', '2022-03-31', '2022-04-30'],
    'feature1': ['10', '21', '30', 'forty'],  # Notice the differences for employee 2
    'feature2': ['100', '200', '300', '400']  # Notice the differences for employee 3
})

# Convert date columns to datetime format
df1['start_date'] = pd.to_datetime(df1['start_date'])
df1['end_date'] = pd.to_datetime(df1['end_date'])
df2['start_date'] = pd.to_datetime(df2['start_date'])
df2['end_date'] = pd.to_datetime(df2['end_date'])

# Merge the DataFrames on keys
merged_df = pd.merge(df1, df2, on=['employee_id', 'start_date', 'end_date'], suffixes=('_df1', '_df2'))

# Initialize dictionaries to store mismatch counts and statistics
mismatch_summary = {}
statistics = {}

# Compare the features
for feature in ['feature1', 'feature2']:
    if merged_df[f'{feature}_df1'].dtype == 'object' or merged_df[f'{feature}_df2'].dtype == 'object':
        # Handle string comparisons
        merged_df[f'{feature}_match'] = merged_df[f'{feature}_df1'] == merged_df[f'{feature}_df2']
        diff_values = merged_df[~merged_df[f'{feature}_match']]
    else:
        # Handle numeric comparisons
        merged_df[f'{feature}_df1'] = pd.to_numeric(merged_df[f'{feature}_df1'], errors='coerce')
        merged_df[f'{feature}_df2'] = pd.to_numeric(merged_df[f'{feature}_df2'], errors='coerce')
        merged_df[f'{feature}_match'] = merged_df[f'{feature}_df1'] == merged_df[f'{feature}_df2']
        merged_df[f'{feature}_diff'] = merged_df[f'{feature}_df1'] - merged_df[f'{feature}_df2']
        diff_values = merged_df[~merged_df[f'{feature}_match']][f'{feature}_diff']
    
    # Count mismatches
    mismatch_summary[feature] = merged_df[~merged_df[f'{feature}_match']]['employee_id'].count()
    
    # Calculate statistics if numeric
    if merged_df[f'{feature}_df1'].dtype in ['int64', 'float64'] and merged_df[f'{feature}_df2'].dtype in ['int64', 'float64']:
        statistics[feature] = {
            'mean_diff': diff_values.mean(),
            'std_diff': diff_values.std(),
            'min_diff': diff_values.min(),
            'max_diff': diff_values.max(),
            'total_diff': diff_values.sum()
        }
    else:
        statistics[feature] = {
            'mean_diff': None,
            'std_diff': None,
            'min_diff': None,
            'max_diff': None,
            'total_diff': None
        }

# Display the mismatch summary
print("Mismatch Summary:")
print(mismatch_summary)

# Display the statistics
print("\nStatistics:")
print(statistics)




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



                import pandas as pd

def rename_and_drop_duplicates(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_x' if i == 0 else dup + '_y' for i in range(sum(cols == dup))]
    df.columns = cols
    y_cols = [col for col in df.columns if col.endswith('_y')]
    df.drop(columns=y_cols, inplace=True)
    return df
    pass
    pass
