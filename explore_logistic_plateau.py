#%%
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import datetime
from typing import List, Optional, Callable

#%%
papers_count_df = pd.read_csv('arxiv_papers_count.csv')
#%%
supremum = st.number_input('Supremum', value=int(1e5))
infinimum = st.number_input('Infinimum', value=500)
logistic_growth = st.number_input('Growth', value=0.35)
logistic_midpoint = st.number_input('Midpoint', value=2023)

#%%
def logistic_func(x):
    return (supremum - infinimum) / (1 + np.exp(-logistic_growth * (x-logistic_midpoint))) + infinimum

x = np.arange(1993, 2100)
y = logistic_func(x)
first_derivative = np.gradient(y)
second_derivative = np.gradient(first_derivative)
third_derivative = np.gradient(second_derivative)
fig = go.Figure()
fig.add_trace(go.Scatter(x=papers_count_df['year'], y=papers_count_df['papers_count'],
                         mode='markers', name='arXiv Papers'))
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Logistic Curve'))
# fig.add_trace(go.Scatter(x=x, y=first_derivative, mode='lines', name='First Derivative', yaxis='y2'))
fig.add_trace(go.Scatter(x=x, y=second_derivative, mode='lines', name='Second Derivative', yaxis='y2'))
fig.add_trace(go.Scatter(x=x, y=third_derivative, mode='lines', name='Third Derivative', yaxis='y2'))

plateau_year = logistic_midpoint+2.5/logistic_growth
# plateau_year = logistic_midpoint+2.5/logistic_growth
# plateau_year - logistic_midpoint = 2.5/logistic_growth
# logistic_growth*(plateau_year - logistic_midpoint) = 2.5
# logistic_growth = 2.5/(plateau_year - logistic_midpoint)

# logistic_midpoint = plateau_year - 2.5/logistic_growth

for vert_line in [logistic_midpoint, plateau_year]:
    fig.add_shape(
        type="line",
        x0=vert_line,
        y0=0,
        x1=vert_line,
        y1=supremum,
        line=dict(
            color="LightSeaGreen",
            width=3,
            dash="dashdot",
        )
    )

fig.update_layout(
    yaxis=dict(title='Logistic Curve'),
    yaxis2=dict(title='Derivatives', overlaying='y', side='right')
)
st.plotly_chart(fig)
#%%
st.write('---')

papers_2010_onward = papers_count_df[papers_count_df['year'] >= 2010]

plateau_year = st.slider('Plateau Year', min_value=2023, max_value=2100, value=2030)
infinimum = 500
def logistic_func(x, supremum, logistic_growth):
    logistic_midpoint = plateau_year - 2.5/logistic_growth
    return (supremum - infinimum) / (1 + np.exp(-logistic_growth * (x-logistic_midpoint))) + infinimum

(supremum, logistic_growth), _ = curve_fit(logistic_func,
                                           papers_2010_onward['year'].values,
                                           papers_2010_onward['papers_count'].values,
                                           p0=(1e5*(plateau_year-2023), .5))

x = np.arange(1993, 2100)
y = logistic_func(x, supremum, logistic_growth)
fig = go.Figure()
fig.add_trace(go.Scatter(x=papers_count_df['year'], y=papers_count_df['papers_count'],
                            mode='markers', name='arXiv Papers'))
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Logistic Curve'))
for vert_line in [plateau_year]:
    fig.add_shape(
        type="line",
        x0=vert_line,
        y0=0,
        x1=vert_line,
        y1=supremum,
        line=dict(
            color="LightSeaGreen",
            width=3,
            dash="dashdot",
        )
    )
st.plotly_chart(fig)
#%%
