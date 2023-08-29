#%%
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from typing import List, Optional, Callable
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.optimize import curve_fit

#%%
papers_count_df = pd.read_csv('arxiv_papers_count.csv')

# fit a logistic curve
def logistic_func(x, supremum, growth, midpoint):
    return supremum / (1 + np.exp(-growth * (x-midpoint)))

x = papers_count_df['year'].values
y = papers_count_df['papers_count'].values
popt, _ = curve_fit(logistic_func, x, y, p0=(1e5, 1, 2020))
print(popt)

supremum = st.slider('Supremum', min_value=0., max_value=1e6, value=popt[0])
growth = st.slider('Growth', min_value=0., max_value=10., value=popt[1])
midpoint = st.slider('Midpoint', min_value=1993., max_value=2053., value=popt[2])

x_pred = np.arange(1993, 2053)
fig = px.scatter(papers_count_df, x='year', y='papers_count',
                 title='Data and Fitted Logistic Curve')
fig.add_scatter(x=x_pred, y=logistic_func(x_pred, supremum, growth, midpoint),
                mode='lines', name='Fitted Curve')

st.plotly_chart(fig)
# %%
st.write("# User-selected supremum")
supremum = st.slider('Supremum', min_value=0., max_value=1e6, value=popt[0], key='user_supremum')

def logistic_func_fixed_sup(x, growth, midpoint):
    return supremum / (1 + np.exp(-growth * (x-midpoint)))

popt, _ = curve_fit(logistic_func_fixed_sup, x, y, p0=(1, 2020))

x_pred = np.arange(1993, 2053)
fig = px.scatter(papers_count_df, x='year', y='papers_count',
                 title='Data and Fitted Logistic Curve')
fig.add_scatter(x=x_pred, y=logistic_func_fixed_sup(x_pred, *popt),
                mode='lines', name='Fitted Curve')
st.plotly_chart(fig)

# %%
