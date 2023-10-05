#%%
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math
from typing import List, Optional, Callable

#%%
current_year = 2023
num_years = st.slider("Number of Years to Predict:", min_value=1, max_value=100, value=30)

#%%
# Polynomial rate function
def logistic_rate(year: int, growth: float, midpoint: float, infinimum: float, supremum: float) -> float:
    return (supremum - infinimum) / (1 + np.exp(-growth * (year - midpoint))) + infinimum

st.title("Exploring the Poisson Simple Function")

# --- First Half for Rate Function ---

st.header("Rate Function Control")

# Logistic coefficients from sliders
growth = st.slider("Growth:", min_value=0.01, max_value=1.0, value=0.35)
midpoint = st.slider("Midpoint:", min_value=2000, max_value=2100, value=2023)
infinimum = st.slider("Infinimum:", min_value=0, max_value=1000, value=500)
supremum = st.slider("Supremum:", min_value=1000, max_value=int(1e5), value=int(1e5))
logistic_args = (growth, midpoint, infinimum, supremum)

# Create x-values for rate function (you can customize this range)
x_vals_rate = list(range(current_year, current_year + num_years + 1))

# Generate y-values for rate function
rate_vals = [logistic_rate(x, *logistic_args) for x in x_vals_rate]

# Create Plotly figure for rate function
fig_rate = go.Figure()
fig_rate.add_trace(go.Scatter(x=x_vals_rate, y=rate_vals, mode='lines+markers'))

fig_rate.update_layout(
    title="Rate Function Visualization",
    xaxis_title="Year",
    yaxis_title="Rate"
)

st.plotly_chart(fig_rate)

#%%
# def poisson_simple(success_years: List[int],
#                    success_rate_func: Optional[Callable[[int], float]],
#                    year_to_predict: int,
#                    min_pred_year: int,
#                    max_pred_year: int,
#                    num_events_predicting: int) -> float:
#     min_year = min(success_years)
#     base_rate_values = [success_rate_func(yr) for yr in range(min_year, max_pred_year+1)]
#     # success_indicator = [1 if yr in success_years else 0 for yr in range(min_year, max_pred_year+1)]
#     # success_rate_values = [base_rate_values[i] * success_indicator[i] for i in range(len(base_rate_values))]
#     # rate_at_year = sum(success_rate_values) / sum(base_rate_values) * success_rate_func(year_to_predict)
#     #! This doesn't depend on the success rate of the years when successes occurred, that seems wrong
#     rate_at_year = len(success_years) / sum(base_rate_values) * success_rate_func(year_to_predict)
#     lambd = rate_at_year * (year_to_predict - min_pred_year)
#     # print(f"{year_to_predict=:<4} {rate_at_year=:.2f} {lambd=:.2f}, {min_pred_year=}")
#     return lambd**num_events_predicting * np.exp(-lambd) / math.factorial(num_events_predicting)

def poisson(success_years: List[int],
            success_rate_func: Optional[Callable[[int], float]],
            num_years: int) -> List[List[float]]:
    # Poi(x_1, ..., x_N | lambda = a s(t) t) = \prod_{i=1}^N (a s(t_i))^x_i exp(-a s(t_i)) / x_i!
    # a = \sum x_i / \sum s(t_i)
    data_range = range(min(success_years), current_year+1)
    rate_scalar = len(success_years) / sum([success_rate_func(yr) for yr in data_range])

    results = []
    for n_shifts in range(num_years):
        shift_results = []
        for yr in range(current_year, current_year+num_years+1):
            lambd = rate_scalar * success_rate_func(yr) * (yr - current_year)
            shift_results.append(lambd**n_shifts * np.exp(-lambd) / math.factorial(n_shifts))

        results.append(shift_results)
    lambd = rate_scalar * success_rate_func(yr) * (yr - current_year)

    return results

st.header("Poisson Distribution Control")

# Streamlit input fields
success_years_input = st.text_input("Success Years (comma-separated):",
                                    "1960, 1982, 1986, 1988, 1995, 2006, 2014, 2017")

# Parsing the success_years_input
success_years = list(map(int, success_years_input.split(',')))

# Create x-values (years to predict)
x_vals = list(range(current_year, current_year + num_years + 1))

# Generate y-values using your function
data = poisson(success_years, lambda year: logistic_rate(year, *logistic_args), num_years)

# Create Plotly figure for poisson_simple
fig = go.Figure()
for n_shifts, shift_data in enumerate(data):
    fig.add_trace(go.Bar(x=x_vals, y=shift_data, name=f"n={n_shifts}"))

fig.update_layout(
    title="Poisson Simple Function Visualization",
    xaxis_title="Year to Predict",
    yaxis_title="Function Value",
    barmode='stack'
)

st.plotly_chart(fig)
#%%
# st.write('---')
# st.header("Poisson Distribution Single-Call")
# def poisson(success_years: List[int],
#             success_rate_func: Optional[Callable[[int], float]],
#             num_years: int) -> List[List[float]]:
#     min_year = min(success_years)
#     mean_success = len(success_years) / (current_year - min_year)
#     base_rate_values = [success_rate_func(yr) for yr in range(min_year, current_year+num_years+1)]
#     mean_rate_val = sum(base_rate_values) / len(base_rate_values)
#     norm_factor = mean_success / mean_rate_val

#     results = []
#     for n_shifts in range(num_years):
#         shift_results = []
#         for year_to_predict in range(current_year, current_year+num_years+1):
#             rate_at_year = norm_factor * success_rate_func(year_to_predict)
#             lambd = rate_at_year * (year_to_predict - current_year)
#             #! This doesn't depend on the success rate of the years when successes occurred, that seems wrong
#             # print(f"{year_to_predict=:<4} {rate_at_year=:.2f} {lambd=:.2f}, {min_pred_year=}")
#             shift_results.append(lambd**n_shifts * np.exp(-lambd) / math.factorial(n_shifts))

#         results.append(shift_results)
#     return results
# num_years = st.slider("Number of Years to Predict:", min_value=1, max_value=100, value=30)
# data = poisson(success_years, lambda year: logistic_rate(year, *logistic_args), num_years)
# fig = go.Figure()
# for n_shifts, shift_data in enumerate(data):
#     fig.add_trace(go.Bar(x=x_vals_poisson, y=shift_data, name=f"n={n_shifts}", showlegend=n_shifts<10))
# fig.update_layout(
#     title="Poisson Distribution Single-Call",
#     xaxis_title="Year to Predict",
#     yaxis_title="Function Value",
#     barmode='stack'
# )
# st.plotly_chart(fig)
#%%
