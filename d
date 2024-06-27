import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example data setup including manager levels
data = {
    'Employee_ID': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
    'Date': ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01'] * 3,
    'Probability': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
    'ec_1': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C'],
    'ec_2': ['X', 'X', 'X', 'X', 'X', 'X', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z'],
    'ec_3': ['P', 'P', 'P', 'P', 'P', 'P', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'R', 'R', 'R', 'R', 'R', 'R']
}

df = pd.DataFrame(data)

# Ensure dates are in datetime format
df['Date'] = pd.to_datetime(df['Date'])

def simulate_employee_attrition(probabilities, num_simulations=1000):
    """Simulate whether an employee attrits each month across multiple simulations."""
    probabilities = np.array(probabilities)
    simulations = np.random.rand(num_simulations, len(probabilities)) < probabilities
    return simulations

def monte_carlo_simulation(df, num_simulations=1000):
    """Run Monte Carlo simulations and store the results for each employee and each month."""
    results = []

    # Group by Employee_ID and collect their monthly probabilities
    grouped = df.groupby('Employee_ID')['Probability'].apply(list)

    for employee_id, probabilities in grouped.items():
        simulation_results = simulate_employee_attrition(probabilities, num_simulations)
        results.append({
            'Employee_ID': employee_id,
            'Simulations': simulation_results
        })

    return results

def aggregate_simulation_results_by_month(results, df, manager_level, num_simulations=1000):
    """Aggregate simulation results by the specified manager level and month."""
    manager_groups = df.groupby(manager_level)
    months = df['Date'].dt.to_period('M').unique()
    aggregated_results = {month: {} for month in months}

    for month in months:
        monthly_df = df[df['Date'].dt.to_period('M') == month]
        manager_monthly_groups = monthly_df.groupby(manager_level)['Employee_ID'].unique()

        for manager, employees in manager_monthly_groups.items():
            if manager not in aggregated_results[month]:
                aggregated_results[month][manager] = np.zeros(num_simulations)

            for employee in employees:
                employee_simulations = next(emp['Simulations'] for emp in results if emp['Employee_ID'] == employee)
                aggregated_results[month][manager] += employee_simulations[:, months.tolist().index(month)]

    return aggregated_results

def compute_monthly_bounds_by_manager(aggregated_results, percentile_lower=2.5, percentile_upper=97.5):
    """Compute the lower and upper bounds for each manager's aggregated simulation results by month."""
    bounds = {}
    for month, managers in aggregated_results.items():
        bounds[month] = {}
        for manager, attritions in managers.items():
            lower_bound = np.percentile(attritions, percentile_lower)
            upper_bound = np.percentile(attritions, percentile_upper)
            bounds[month][manager] = (lower_bound, upper_bound)
    return bounds

def compute_actual_attrition(df, manager_level):
    """Compute the actual attrition based on the probabilities for each manager by month."""
    actual_attrition = df.groupby([df['Date'].dt.to_period('M'), manager_level])['Probability'].sum().unstack(fill_value=0)
    return actual_attrition

# Run the simulation
num_simulations = 1000
results = monte_carlo_simulation(df, num_simulations)

# Aggregate results by each manager level and month
aggregated_ec_1 = aggregate_simulation_results_by_month(results, df, 'ec_1', num_simulations)
aggregated_ec_2 = aggregate_simulation_results_by_month(results, df, 'ec_2', num_simulations)
aggregated_ec_3 = aggregate_simulation_results_by_month(results, df, 'ec_3', num_simulations)

# Compute monthly bounds for each manager level
monthly_bounds_ec_1 = compute_monthly_bounds_by_manager(aggregated_ec_1)
monthly_bounds_ec_2 = compute_monthly_bounds_by_manager(aggregated_ec_2)
monthly_bounds_ec_3 = compute_monthly_bounds_by_manager(aggregated_ec_3)

# Compute actual attrition for each manager level by month
actual_attrition_ec_1 = compute_actual_attrition(df, 'ec_1')
actual_attrition_ec_2 = compute_actual_attrition(df, 'ec_2')
actual_attrition_ec_3 = compute_actual_attrition(df, 'ec_3')

# Display the monthly bounds for each manager level along with actual attrition
def display_monthly_bounds_and_actual(bounds, actual, manager_level):
    print(f"Monthly Bounds and Actual Attrition for {manager_level} managers:")
    for month in bounds:
        print(f"\nMonth: {month}")
        for manager, (lower, upper) in bounds[month].items():
            actual_value = actual.loc[month, manager] if manager in actual.loc[month] else 0
            print(f"Manager {manager}: Lower Bound = {lower}, Upper Bound = {upper}, Actual Attrition = {actual_value}")

display_monthly_bounds_and_actual(monthly_bounds_ec_1, actual_attrition_ec_1, 'ec_1')
display_monthly_bounds_and_actual(monthly_bounds_ec_2, actual_attrition_ec_2, 'ec_2')
display_monthly_bounds_and_actual(monthly_bounds_ec_3, actual_attrition_ec_3, 'ec_3')

# Plot results for one manager level and one month as an example (ec_1, January)
month_to_plot = '2024-01'
manager_to_plot = list(aggregated_ec_1[month_to_plot].keys())[0]  # Plot for the first manager in ec_1 for January
plt.hist(aggregated_ec_1[month_to_plot][manager_to_plot], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Total Attritions')
plt.ylabel('Frequency')
plt.title(f'Monte Carlo Simulation of Total Employee Attrition for Manager {manager_to_plot} in {month_to_plot}')
plt.axvline(monthly_bounds_ec_1[month_to_plot][manager_to_plot][0], color='r', linestyle='dashed', linewidth=1)
plt.axvline(monthly_bounds_ec_1[month_to_plot][manager_to_plot][1], color='r', linestyle='dashed', linewidth=1)
actual_value = actual_attrition_ec_1.loc[month_to_plot, manager_to_plot] if manager_to_plot in actual_attrition_ec_1.loc[month_to_plot] else 0
plt.axvline(actual_value, color='g', linestyle='solid', linewidth=2)
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example data setup including manager levels
data = {
    'Employee_ID': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
    'Date': ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01'] * 3,
    'Probability': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
    'ec_1': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C'],
    'ec_2': ['X', 'X', 'X', 'X', 'X', 'X', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z'],
    'ec_3': ['P', 'P', 'P', 'P', 'P', 'P', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'R', 'R', 'R', 'R', 'R', 'R']
}

df = pd.DataFrame(data)

# Ensure dates are in datetime format
df['Date'] = pd.to_datetime(df['Date'])

def simulate_employee_attrition(probabilities, num_simulations=1000):
    """Simulate whether an employee attrits each month across multiple simulations."""
    probabilities = np.array(probabilities)
    simulations = np.random.rand(num_simulations, len(probabilities)) < probabilities
    return simulations

def monte_carlo_simulation(df, num_simulations=1000):
    """Run Monte Carlo simulations and store the results for each employee and each month."""
    results = []

    # Group by Employee_ID and collect their monthly probabilities
    grouped = df.groupby('Employee_ID')['Probability'].apply(list)

    for employee_id, probabilities in grouped.items():
        simulation_results = simulate_employee_attrition(probabilities, num_simulations)
        results.append({
            'Employee_ID': employee_id,
            'Simulations': simulation_results
        })

    return results

def aggregate_simulation_results_by_month(results, df, manager_level, num_simulations=1000):
    """Aggregate simulation results by the specified manager level and month."""
    manager_groups = df.groupby(manager_level)
    months = df['Date'].dt.to_period('M').unique()
    aggregated_results = {month: {} for month in months}

    for month in months:
        monthly_df = df[df['Date'].dt.to_period('M') == month]
        manager_monthly_groups = monthly_df.groupby(manager_level)['Employee_ID'].unique()

        for manager, employees in manager_monthly_groups.items():
            if manager not in aggregated_results[month]:
                aggregated_results[month][manager] = np.zeros(num_simulations)

            for employee in employees:
                employee_simulations = next(emp['Simulations'] for emp in results if emp['Employee_ID'] == employee)
                aggregated_results[month][manager] += employee_simulations[:, months.tolist().index(month)]

    return aggregated_results

def compute_monthly_bounds_by_manager(aggregated_results, percentile_lower=2.5, percentile_upper=97.5):
    """Compute the lower and upper bounds for each manager's aggregated simulation results by month."""
    bounds = {}
    for month, managers in aggregated_results.items():
        bounds[month] = {}
        for manager, attritions in managers.items():
            lower_bound = np.percentile(attritions, percentile_lower)
            upper_bound = np.percentile(attritions, percentile_upper)
            bounds[month][manager] = (lower_bound, upper_bound)
    return bounds

# Run the simulation
num_simulations = 1000
results = monte_carlo_simulation(df, num_simulations)

# Aggregate results by each manager level and month
aggregated_ec_1 = aggregate_simulation_results_by_month(results, df, 'ec_1', num_simulations)
aggregated_ec_2 = aggregate_simulation_results_by_month(results, df, 'ec_2', num_simulations)
aggregated_ec_3 = aggregate_simulation_results_by_month(results, df, 'ec_3', num_simulations)

# Compute monthly bounds for each manager level
monthly_bounds_ec_1 = compute_monthly_bounds_by_manager(aggregated_ec_1)
monthly_bounds_ec_2 = compute_monthly_bounds_by_manager(aggregated_ec_2)
monthly_bounds_ec_3 = compute_monthly_bounds_by_manager(aggregated_ec_3)

# Display the monthly bounds for each manager level
def display_monthly_bounds(bounds, manager_level):
    print(f"Monthly Bounds for {manager_level} managers:")
    for month, managers in bounds.items():
        print(f"\nMonth: {month}")
        for manager, (lower, upper) in managers.items():
            print(f"Manager {manager}: Lower Bound = {lower}, Upper Bound = {upper}")

display_monthly_bounds(monthly_bounds_ec_1, 'ec_1')
display_monthly_bounds(monthly_bounds_ec_2, 'ec_2')
display_monthly_bounds(monthly_bounds_ec_3, 'ec_3')

# Plot results for one manager level and one month as an example (ec_1, January)
month_to_plot = '2024-01'
manager_to_plot = list(aggregated_ec_1[month_to_plot].keys())[0]  # Plot for the first manager in ec_1 for January
plt.hist(aggregated_ec_1[month_to_plot][manager_to_plot], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Total Attritions')
plt.ylabel('Frequency')
plt.title(f'Monte Carlo Simulation of Total Employee Attrition for Manager {manager_to_plot} in {month_to_plot}')
plt.axvline(monthly_bounds_ec_1[month_to_plot][manager_to_plot][0], color='r', linestyle='dashed', linewidth=1)
plt.axvline(monthly_bounds_ec_1[month_to_plot][manager_to_plot][1], color='r', linestyle='dashed', linewidth=1)
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example data setup including manager levels
data = {
    'Employee_ID': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
    'Date': ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01'] * 3,
    'Probability': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
    'ec_1': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C'],
    'ec_2': ['X', 'X', 'X', 'X', 'X', 'X', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z'],
    'ec_3': ['P', 'P', 'P', 'P', 'P', 'P', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'R', 'R', 'R', 'R', 'R', 'R']
}

df = pd.DataFrame(data)

# Ensure dates are in datetime format
df['Date'] = pd.to_datetime(df['Date'])

def simulate_employee_attrition(probabilities, num_simulations=1000):
    """Simulate whether an employee attrits each month across multiple simulations."""
    probabilities = np.array(probabilities)
    simulations = np.random.rand(num_simulations, len(probabilities)) < probabilities
    return simulations

def monte_carlo_simulation(df, num_simulations=1000):
    """Run Monte Carlo simulations and store the results for each employee and each month."""
    results = []

    # Group by Employee_ID and collect their monthly probabilities
    grouped = df.groupby('Employee_ID')['Probability'].apply(list)

    for employee_id, probabilities in grouped.items():
        simulation_results = simulate_employee_attrition(probabilities, num_simulations)
        results.append({
            'Employee_ID': employee_id,
            'Simulations': simulation_results
        })

    return results

def aggregate_simulation_results_by_month(results, df, manager_level, num_simulations=1000):
    """Aggregate simulation results by the specified manager level and month."""
    manager_groups = df.groupby(manager_level)
    months = df['Date'].dt.to_period('M').unique()
    aggregated_results = {month: {} for month in months}

    for month in months:
        monthly_df = df[df['Date'].dt.to_period('M') == month]
        manager_monthly_groups = monthly_df.groupby(manager_level)['Employee_ID'].unique()

        for manager, employees in manager_monthly_groups.items():
            if manager not in aggregated_results[month]:
                aggregated_results[month][manager] = np.zeros(num_simulations)

            for employee in employees:
                employee_simulations = next(emp['Simulations'] for emp in results if emp['Employee_ID'] == employee)
                aggregated_results[month][manager] += employee_simulations[:, months.tolist().index(month)]

    return aggregated_results

def compute_monthly_bounds(aggregated_results, percentile_lower=2.5, percentile_upper=97.5):
    """Compute the lower and upper bounds for the aggregated simulation results by month."""
    bounds = {}
    for month, managers in aggregated_results.items():
        total_attritions = np.zeros_like(list(managers.values())[0])
        for attritions in managers.values():
            total_attritions += attritions
        lower_bound = np.percentile(total_attritions, percentile_lower)
        upper_bound = np.percentile(total_attritions, percentile_upper)
        bounds[month] = (lower_bound, upper_bound)
    return bounds

# Run the simulation
num_simulations = 1000
results = monte_carlo_simulation(df, num_simulations)

# Aggregate results by each manager level and month
aggregated_ec_1 = aggregate_simulation_results_by_month(results, df, 'ec_1', num_simulations)
aggregated_ec_2 = aggregate_simulation_results_by_month(results, df, 'ec_2', num_simulations)
aggregated_ec_3 = aggregate_simulation_results_by_month(results, df, 'ec_3', num_simulations)

# Compute monthly bounds for each manager level
monthly_bounds_ec_1 = compute_monthly_bounds(aggregated_ec_1)
monthly_bounds_ec_2 = compute_monthly_bounds(aggregated_ec_2)
monthly_bounds_ec_3 = compute_monthly_bounds(aggregated_ec_3)

# Display the monthly bounds for each manager level
def display_monthly_bounds(bounds, manager_level):
    print(f"Monthly Bounds for {manager_level} managers:")
    for month, (lower, upper) in bounds.items():
        print(f"Month: {month} -> Lower Bound = {lower}, Upper Bound = {upper}")

display_monthly_bounds(monthly_bounds_ec_1, 'ec_1')
display_monthly_bounds(monthly_bounds_ec_2, 'ec_2')
display_monthly_bounds(monthly_bounds_ec_3, 'ec_3')

# Plot results for one manager level as an example (ec_1)
for month in monthly_bounds_ec_1:
    total_attritions = np.zeros(num_simulations)
    for attritions in aggregated_ec_1[month].values():
        total_attritions += attritions
    plt.hist(total_attritions, bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel('Total Attritions')
    plt.ylabel('Frequency')
    plt.title(f'Monte Carlo Simulation of Total Employee Attrition for {month}')
    plt.axvline(monthly_bounds_ec_1[month][0], color='r', linestyle='dashed', linewidth=1)
    plt.axvline(monthly_bounds_ec_1[month][1], color='r', linestyle='dashed', linewidth=1)
    plt.show()













import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example data setup including manager levels
data = {
    'Employee_ID': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
    'Date': ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01'] * 3,
    'Probability': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
    'ec_1': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C'],
    'ec_2': ['X', 'X', 'X', 'X', 'X', 'X', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z'],
    'ec_3': ['P', 'P', 'P', 'P', 'P', 'P', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'R', 'R', 'R', 'R', 'R', 'R']
}

df = pd.DataFrame(data)

def simulate_employee_attrition(probabilities, num_simulations=1000):
    """Simulate whether an employee attrits each month across multiple simulations."""
    probabilities = np.array(probabilities)
    simulations = np.random.rand(num_simulations, len(probabilities)) < probabilities
    return simulations

def monte_carlo_simulation(df, num_simulations=1000):
    """Run Monte Carlo simulations and store the results for each employee and each month."""
    results = []

    # Group by Employee_ID and collect their monthly probabilities
    grouped = df.groupby('Employee_ID')['Probability'].apply(list)

    for employee_id, probabilities in grouped.items():
        simulation_results = simulate_employee_attrition(probabilities, num_simulations)
        results.append({
            'Employee_ID': employee_id,
            'Simulations': simulation_results
        })

    return results

def aggregate_simulation_results_by_month(results, df, manager_level, num_simulations=1000):
    """Aggregate simulation results by the specified manager level and month."""
    manager_groups = df.groupby(manager_level)
    months = df['Date'].unique()
    aggregated_results = {month: {} for month in months}

    for month in months:
        monthly_df = df[df['Date'] == month]
        manager_monthly_groups = monthly_df.groupby(manager_level)['Employee_ID'].unique()

        for manager, employees in manager_monthly_groups.items():
            if manager not in aggregated_results[month]:
                aggregated_results[month][manager] = np.zeros(num_simulations)

            for employee in employees:
                employee_simulations = next(emp['Simulations'] for emp in results if emp['Employee_ID'] == employee)
                aggregated_results[month][manager] += employee_simulations[:, months.tolist().index(month)]

    return aggregated_results

def compute_bounds(aggregated_results, percentile_lower=2.5, percentile_upper=97.5):
    """Compute the lower and upper bounds for the aggregated simulation results by month."""
    bounds = {month: {} for month in aggregated_results.keys()}
    for month, managers in aggregated_results.items():
        for manager, attritions in managers.items():
            lower_bound = np.percentile(attritions, percentile_lower)
            upper_bound = np.percentile(attritions, percentile_upper)
            bounds[month][manager] = (lower_bound, upper_bound)
    return bounds

# Run the simulation
num_simulations = 1000
results = monte_carlo_simulation(df, num_simulations)

# Aggregate results by each manager level and month
aggregated_ec_1 = aggregate_simulation_results_by_month(results, df, 'ec_1', num_simulations)
aggregated_ec_2 = aggregate_simulation_results_by_month(results, df, 'ec_2', num_simulations)
aggregated_ec_3 = aggregate_simulation_results_by_month(results, df, 'ec_3', num_simulations)

# Compute bounds for each manager level and month
bounds_ec_1 = compute_bounds(aggregated_ec_1)
bounds_ec_2 = compute_bounds(aggregated_ec_2)
bounds_ec_3 = compute_bounds(aggregated_ec_3)

# Display the bounds for each manager level and month
def display_bounds(bounds, manager_level):
    print(f"Bounds for {manager_level} managers:")
    for month, managers in bounds.items():
        print(f"\nMonth: {month}")
        for manager, (lower, upper) in managers.items():
            print(f"Manager {manager}: Lower Bound = {lower}, Upper Bound = {upper}")

display_bounds(bounds_ec_1, 'ec_1')
display_bounds(bounds_ec_2, 'ec_2')
display_bounds(bounds_ec_3, 'ec_3')

# Plot results for one manager level and one month as an example (ec_1, January)
manager_to_plot = list(aggregated_ec_1['2024-01-01'].keys())[0]  # Plot for the first manager in ec_1 for January
plt.hist(aggregated_ec_1['2024-01-01'][manager_to_plot], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Total Attritions')
plt.ylabel('Frequency')
plt.title(f'Monte Carlo Simulation of Total Employee Attrition for Manager {manager_to_plot} in January')
plt.axvline(bounds_ec_1['2024-01-01'][manager_to_plot][0], color='r', linestyle='dashed', linewidth=1)
plt.axvline(bounds_ec_1['2024-01-01'][manager_to_plot][1], color='r', linestyle='dashed', linewidth=1)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example data setup including manager levels
data = {
    'Employee_ID': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
    'Date': ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01'] * 3,
    'Probability': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
    'ec_1': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C'],
    'ec_2': ['X', 'X', 'X', 'X', 'X', 'X', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z'],
    'ec_3': ['P', 'P', 'P', 'P', 'P', 'P', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'R', 'R', 'R', 'R', 'R', 'R']
}

df = pd.DataFrame(data)

def simulate_employee_attrition(probabilities, num_simulations=1000):
    """Simulate whether an employee attrits each month across multiple simulations."""
    probabilities = np.array(probabilities)
    simulations = np.random.rand(num_simulations, len(probabilities)) < probabilities
    return simulations

def monte_carlo_simulation(df, num_simulations=1000):
    """Run Monte Carlo simulations and store the results for each employee and each month."""
    results = []

    # Group by Employee_ID and collect their monthly probabilities
    grouped = df.groupby('Employee_ID')['Probability'].apply(list)

    for employee_id, probabilities in grouped.items():
        simulation_results = simulate_employee_attrition(probabilities, num_simulations)
        results.append({
            'Employee_ID': employee_id,
            'Simulations': simulation_results
        })

    return results

def aggregate_simulation_results_by_manager(results, df, manager_level, num_simulations=1000):
    """Aggregate simulation results by the specified manager level."""
    manager_groups = df.groupby(manager_level)['Employee_ID'].unique()
    aggregated_results = {}

    for manager, employees in manager_groups.items():
        total_attritions = np.zeros(num_simulations)
        for employee in employees:
            employee_simulations = next(emp['Simulations'] for emp in results if emp['Employee_ID'] == employee)
            total_attritions += employee_simulations.sum(axis=1)
        aggregated_results[manager] = total_attritions

    return aggregated_results

def compute_bounds(aggregated_results, percentile_lower=2.5, percentile_upper=97.5):
    """Compute the lower and upper bounds for the aggregated simulation results."""
    bounds = {}
    for manager, attritions in aggregated_results.items():
        lower_bound = np.percentile(attritions, percentile_lower)
        upper_bound = np.percentile(attritions, percentile_upper)
        bounds[manager] = (lower_bound, upper_bound)
    return bounds

# Run the simulation
results = monte_carlo_simulation(df)

# Aggregate results by each manager level
aggregated_ec_1 = aggregate_simulation_results_by_manager(results, df, 'ec_1')
aggregated_ec_2 = aggregate_simulation_results_by_manager(results, df, 'ec_2')
aggregated_ec_3 = aggregate_simulation_results_by_manager(results, df, 'ec_3')

# Compute bounds for each manager level
bounds_ec_1 = compute_bounds(aggregated_ec_1)
bounds_ec_2 = compute_bounds(aggregated_ec_2)
bounds_ec_3 = compute_bounds(aggregated_ec_3)

# Display the bounds for each manager level
print("Bounds for ec_1 managers:")
for manager, (lower, upper) in bounds_ec_1.items():
    print(f"Manager {manager}: Lower Bound = {lower}, Upper Bound = {upper}")

print("\nBounds for ec_2 managers:")
for manager, (lower, upper) in bounds_ec_2.items():
    print(f"Manager {manager}: Lower Bound = {lower}, Upper Bound = {upper}")

print("\nBounds for ec_3 managers:")
for manager, (lower, upper) in bounds_ec_3.items():
    print(f"Manager {manager}: Lower Bound = {lower}, Upper Bound = {upper}")

# Plot results for one manager level as an example (ec_1)
manager_to_plot = list(aggregated_ec_1.keys())[0]  # Plot for the first manager in ec_1
plt.hist(aggregated_ec_1[manager_to_plot], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Total Attritions')
plt.ylabel('Frequency')
plt.title(f'Monte Carlo Simulation of Total Employee Attrition for Manager {manager_to_plot}')
plt.axvline(bounds_ec_1[manager_to_plot][0], color='r', linestyle='dashed', linewidth=1)
plt.axvline(bounds_ec_1[manager_to_plot][1], color='r', linestyle='dashed', linewidth=1)
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example data setup including manager levels
data = {
    'Employee_ID': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
    'Date': ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01'] * 3,
    'Probability': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
    'ec_1': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C'],
    'ec_2': ['X', 'X', 'X', 'X', 'X', 'X', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z'],
    'ec_3': ['P', 'P', 'P', 'P', 'P', 'P', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'R', 'R', 'R', 'R', 'R', 'R']
}

df = pd.DataFrame(data)

def simulate_employee_attrition(probabilities, num_simulations=1000):
    """Simulate whether an employee attrits each month across multiple simulations."""
    simulation_results = {month: [] for month in range(len(probabilities))}
    for _ in range(num_simulations):
        for month, prob in enumerate(probabilities):
            simulation_results[month].append(int(np.random.rand() < prob))
    return simulation_results

def monte_carlo_simulation(df, num_simulations=1000):
    """Run Monte Carlo simulations and store the results for each employee and each month."""
    results = []

    # Group by Employee_ID and collect their monthly probabilities
    grouped = df.groupby('Employee_ID')['Probability'].apply(list)

    for employee_id, probabilities in grouped.items():
        employee_results = {'Employee_ID': employee_id}
        simulation_results = simulate_employee_attrition(probabilities, num_simulations)
        for month, month_results in simulation_results.items():
            employee_results[f'Month_{month+1}'] = month_results
        results.append(employee_results)

    return results

# Run the simulation
results = monte_carlo_simulation(df)

def aggregate_simulation_results_by_manager(results, df, manager_level, num_simulations=1000):
    """Aggregate simulation results by the specified manager level."""
    manager_groups = df.groupby(manager_level)['Employee_ID'].unique()
    aggregated_results = {}

    for manager, employees in manager_groups.items():
        total_attritions = np.zeros(num_simulations)
        for simulation_idx in range(num_simulations):
            for employee in employees:
                employee_results = next(emp for emp in results if emp['Employee_ID'] == employee)
                for month in range(6):  # Assuming 6 months
                    total_attritions[simulation_idx] += employee_results[f'Month_{month+1}'][simulation_idx]
        aggregated_results[manager] = total_attritions

    return aggregated_results

def compute_bounds(aggregated_results, percentile_lower=2.5, percentile_upper=97.5):
    """Compute the lower and upper bounds for the aggregated simulation results."""
    bounds = {}
    for manager, attritions in aggregated_results.items():
        lower_bound = np.percentile(attritions, percentile_lower)
        upper_bound = np.percentile(attritions, percentile_upper)
        bounds[manager] = (lower_bound, upper_bound)
    return bounds

# Aggregate results by each manager level
aggregated_ec_1 = aggregate_simulation_results_by_manager(results, df, 'ec_1')
aggregated_ec_2 = aggregate_simulation_results_by_manager(results, df, 'ec_2')
aggregated_ec_3 = aggregate_simulation_results_by_manager(results, df, 'ec_3')

# Compute bounds for each manager level
bounds_ec_1 = compute_bounds(aggregated_ec_1)
bounds_ec_2 = compute_bounds(aggregated_ec_2)
bounds_ec_3 = compute_bounds(aggregated_ec_3)

# Display the bounds for each manager level
print("Bounds for ec_1 managers:")
for manager, (lower, upper) in bounds_ec_1.items():
    print(f"Manager {manager}: Lower Bound = {lower}, Upper Bound = {upper}")

print("\nBounds for ec_2 managers:")
for manager, (lower, upper) in bounds_ec_2.items():
    print(f"Manager {manager}: Lower Bound = {lower}, Upper Bound = {upper}")

print("\nBounds for ec_3 managers:")
for manager, (lower, upper) in bounds_ec_3.items():
    print(f"Manager {manager}: Lower Bound = {lower}, Upper Bound = {upper}")

# Plot results for one manager level as an example (ec_1)
manager_to_plot = list(aggregated_ec_1.keys())[0]  # Plot for the first manager in ec_1
plt.hist(aggregated_ec_1[manager_to_plot], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Total Attritions')
plt.ylabel('Frequency')
plt.title(f'Monte Carlo Simulation of Total Employee Attrition for Manager {manager_to_plot}')
plt.axvline(bounds_ec_1[manager_to_plot][0], color='r', linestyle='dashed', linewidth=1)
plt.axvline(bounds_ec_1[manager_to_plot][1], color='r', linestyle='dashed', linewidth=1)
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example data setup
data = {
    'Employee_ID': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
    'Date': ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01'] * 3,
    'Probability': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
}

df = pd.DataFrame(data)

def simulate_employee_attrition(probabilities, num_simulations=1000):
    """Simulate whether an employee attrits each month across multiple simulations."""
    simulation_results = {month: [] for month in range(len(probabilities))}
    for _ in range(num_simulations):
        for month, prob in enumerate(probabilities):
            simulation_results[month].append(int(np.random.rand() < prob))
    return simulation_results

def monte_carlo_simulation(df, num_simulations=1000):
    """Run Monte Carlo simulations and store the results for each employee and each month."""
    results = []

    # Group by Employee_ID and collect their monthly probabilities
    grouped = df.groupby('Employee_ID')['Probability'].apply(list)

    for employee_id, probabilities in grouped.items():
        employee_results = {'Employee_ID': employee_id}
        simulation_results = simulate_employee_attrition(probabilities, num_simulations)
        for month, month_results in simulation_results.items():
            employee_results[f'Month_{month+1}'] = month_results
        results.append(employee_results)

    return results

# Run the simulation
results = monte_carlo_simulation(df)

# Aggregate simulation results
total_attritions = []
for simulation_idx in range(1000):  # Assuming 1000 simulations
    total_attrition = 0
    for employee in results:
        for month in range(6):  # Assuming 6 months
            total_attrition += employee[f'Month_{month+1}'][simulation_idx]
    total_attritions.append(total_attrition)

# Compute summary statistics
lower_bound = np.percentile(total_attritions, 2.5)
upper_bound = np.percentile(total_attritions, 97.5)
mean_attrition = np.mean(total_attritions)
std_attrition = np.std(total_attritions)

print(f"Mean total attrition: {mean_attrition:.2f}")
print(f"Standard deviation of total attrition: {std_attrition:.2f}")
print(f"95% confidence interval: ({lower_bound}, {upper_bound})")

# Plot histogram of total attritions
plt.hist(total_attritions, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Total Attritions Over 6 Months')
plt.ylabel('Frequency')
plt.title('Monte Carlo Simulation of Total Employee Attrition')
plt.axvline(lower_bound, color='r', linestyle='dashed', linewidth=1)
plt.axvline(upper_bound, color='r', linestyle='dashed', linewidth=1)
plt.show()
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
