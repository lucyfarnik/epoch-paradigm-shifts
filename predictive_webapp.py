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

         Developed by [Lucy Farnik](https://www.linkedin.com/in/lucy-farnik/),
         [Francesca Sheeka](https://www.linkedin.com/in/fsheeka/),
         and [Pablo Villalobos](https://www.linkedin.com/in/pablo-villalobos-sanchez/).
         """)

#%%
if 'paradigm_shifts' not in st.session_state:
    st.session_state.paradigm_shifts = [ #TODO: add more, fact check years
        'Backpropagation (1960)',
        'Hopfield networks (1982)',
        'Recurrent Neural Networks (1986)',
        'Convolutional Neural Networks (1988)',
        'Support Vector Machines (1995)',
        'Deep Learning (2006)',
        'Generative Adversarial Networks (2014)',
        'Transformers (2017)',
        'Foundation models (2020)'
    ]


# ability to add custom paradigm shifts
col1, col2 = st.columns([5, 1])
with col1:
    custom_paradigm_shift = st.text_input('Add custom paradigm shift (then press the Add button)')
with col2:
    st.write('## ')
    if st.button('Add'):
        try:
            year = int(custom_paradigm_shift[-5:-1])
            st.session_state.paradigm_shifts.append(custom_paradigm_shift)
        except ValueError:
            st.error('Please enter a valid year in the brackets')

selected_years = []

# list of checkboxes
for paradigm_shift in st.session_state.paradigm_shifts:
    if st.checkbox(paradigm_shift, value=True, key=paradigm_shift):
        selected_years.append(int(paradigm_shift[-5:-1]))

#TODO: add ability to add custom paradigm shifts
#TODO: let people rate the impact of paradigm shifts; adapt the model accordingly

st.write('---')
#%%
def get_prediction_dist(selected_years: List[int], num_yrs_forward: int = 20) -> List[float]:
    # prob of no success during t time = (1+t/T)^{-S} (T = time since first paradigm shift, S = number of paradigm shifts)
    current_year = datetime.datetime.now().year
    first_shift = min(selected_years)
    sample_time_period = current_year - first_shift # T in the formula above
    n_shifts = len(selected_years) # S in the formula above

    # compute the probabilities
    prob_shift_cumulative = [1 - (1+i/sample_time_period)**(-n_shifts)
                             for i in range(1, num_yrs_forward+1)]
    prob_shift = [prob_shift_cumulative[0]] + [
        prob_shift_cumulative[i] - prob_shift_cumulative[i-1]
        for i in range(1, num_yrs_forward)]
    
    # distribution over the next num_yrs_forward years, dummy dist for now
    return pd.DataFrame({
        'Year': range(current_year, current_year+num_yrs_forward),
        'Probability': [round(p, 3) for p in prob_shift], # round to 3 decimal places
        'Cumulative Probability': [round(p, 3) for p in prob_shift_cumulative],
    })


#%%
# Chart options
col1, col2 = st.columns(2)
with col1:
    show_cumulative = st.checkbox('Show cumulative probability')
with col2:
    num_yrs_forward = st.slider('Number of years to predict', 1, 100, 20)

#%%
# get the data
data = get_prediction_dist(selected_years, num_yrs_forward)

#%%
# compute when the cumulative probabilities surpass certain thresholds 
cummulative_threshs = []
for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
    try:
        cummulative_threshs.append((p, data[data['Cumulative Probability'] >= p]['Year'].iloc[0]))
    except IndexError:
        pass
#%%
# Create the plot using Plotly
fig = go.Figure()

# Add a bar chart for yearly probabilities
fig.add_trace(go.Bar(x=data['Year'], y=data['Probability'], name='Probability'))

if show_cumulative:
    # Add a line chart for cumulative probabilities
    fig.add_trace(go.Scatter(x=data['Year'], y=data['Cumulative Probability'],
                             mode='lines', name='Cumulative Probability'))

chart_height = 1 if show_cumulative else data['Probability'].max()
for (prob, year) in cummulative_threshs:
    fig.add_shape(
        type='line',
        yref='y', y0=0, y1=chart_height,
        xref='x', x0=year, x1=year,
        line=dict(color='red', width=2, dash='dot'),
    )
    fig.add_trace(go.Scatter(
        x=[year],
        y=[chart_height+0.005],
        text=[f"{100*prob:.0f}%"],
        mode="text",
        showlegend=False
    ))

# Render the Plotly chart in Streamlit
st.plotly_chart(fig)

#%%
# print the year when the cumulative probability of a paradigm shift reach thresholds
st.write('\n'.join([f"- {p*100}% chance of a paradigm shift by {y}" for p, y in cummulative_threshs]))

#%%