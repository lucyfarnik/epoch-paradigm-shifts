#%%
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import datetime
from typing import List

#%%
st.write("""
         # Predictive model of future paradigm shifts
         Select what you consider to be a paradigm shift in the history of ML.
         The app will give you a prediction of when you can expect to see the next
         paradigm shift, based on the [time-invariant Laplace's rule](https://epochai.org/blog/a-time-invariant-version-of-laplace-s-rule).
         """)

#%%
potential_paradigm_shifts = [ #TODO: add more
    'Transformers (2017)',
    'Generative Adversarial Networks (2014)',
    'Deep Learning (2006)',
    'Support Vector Machines (1995)',
    'Backpropagation (1986)',
]

selected_years = []

# list of checkboxes
for i, paradigm_shift in enumerate(potential_paradigm_shifts):
    if st.checkbox(paradigm_shift, value=True, key=i):
        selected_years.append(int(paradigm_shift[-5:-1]))
#TODO: add ability to add custom paradigm shifts
#TODO: let people rate the impact of paradigm shifts; adapt the model accordingly

#%%
def get_prediction_dist(selected_years: List[int], num_yrs_forward: int = 20) -> List[float]:
    # prob of no success during t time = (1+t/T)^{-S} (T = time since first paradigm shift, S = number of paradigm shifts)
    current_year = datetime.datetime.now().year
    first_shift = min(selected_years)
    sample_time_period = current_year - first_shift # T in the formula above
    n_shifts = len(selected_years) # S in the formula above

    # compute the probabilities
    prob_shift_cumulative = [1 - (1+i/sample_time_period)**(-n_shifts)
                             for i in range(num_yrs_forward)]
    prob_shift = [prob_shift_cumulative[0]] + [
        prob_shift_cumulative[i] - prob_shift_cumulative[i-1]
        for i in range(1, num_yrs_forward)]
    
    # distribution over the next num_yrs_forward years, dummy dist for now
    return pd.DataFrame({
        'Year': range(current_year, current_year+num_yrs_forward),
        'Probability': [round(p, 2) for p in prob_shift], # round to 2 decimal places
        'Cumulative Probability': [round(p, 2) for p in prob_shift_cumulative],
    })

#%%
data = get_prediction_dist(selected_years)

# Create the plot using Plotly
fig = go.Figure()

# Add a bar chart for yearly probabilities
fig.add_trace(go.Bar(x=data['Year'], y=data['Probability'], name='Probability'))

# Add a line chart for cumulative probabilities
fig.add_trace(go.Scatter(x=data['Year'], y=data['Cumulative Probability'], mode='lines', name='Cumulative Probability'))

# Render the Plotly chart in Streamlit
st.plotly_chart(fig)

#%%