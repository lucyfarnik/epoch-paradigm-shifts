#%%
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import datetime
from typing import List, Optional, Callable

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

#TODO: let people rate the impact of paradigm shifts; adapt the model accordingly

#%%
num_yrs_forward = st.slider('Number of years to predict', 10, 100, 30, 5)
st.write('---')
#%%
current_year = datetime.datetime.now().year
def get_prediction_dist(selected_years: List[int], num_yrs_forward: int = 30) -> List[float]:
    # prob of no success during t time = (1+t/T)^{-S} (T = time since first paradigm shift, S = number of paradigm shifts)
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
st.write("## Next paradigm shift date")
# Chart options
show_cumulative = st.checkbox('Show cumulative probability')

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
# if there are multiple cumulative probability values in the same year,
# only keep the highest one so that they don't visually overlap
cummulative_threshs = [cummulative_threshs[i]
                       for i in range(len(cummulative_threshs))
                       if i==len(cummulative_threshs)-1 or
                            cummulative_threshs[i][1] != cummulative_threshs[i+1][1]]

#%%
# Create the plot using Plotly
fig = go.Figure()

# Add a bar chart for yearly probabilities
fig.add_trace(go.Bar(x=data['Year'], y=data['Probability'],
                     name='Probability', showlegend=show_cumulative))

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
# show number of paradigm shifts expected by a given year
num_successes = len(selected_years)
num_years = current_year - min(selected_years)
st.write("""
         ---
         ## Number of paradigm shifts predicted by a given year
""")
num_shifts_by_year = [round(t*num_successes/num_years, 2)
                      for t in range(num_yrs_forward)]

#%%
# print the year when the predicted number of paradigm shifts crosses integer thresholds
num_shifts_thresh_years = []
num_shifts_thresh = 1
for year_offset, num_shifts in enumerate(num_shifts_by_year):
    if num_shifts >= num_shifts_thresh:
        num_shifts_thresh_years.append((num_shifts_thresh, current_year + year_offset))
        num_shifts_thresh += 1

st.write('\n'.join([f"- {n} paradigm shifts predicted by {y}" for n, y in num_shifts_thresh_years]))
# %%
st.write("## Probability of seeing a given number of paradigm shifts by year")

def laplace_rule_succ(num_successes: int, num_trials: int) -> float:
    """
    Probability of seeing a new paradigm shift in the next year, given that we've
    seen num_successes paradigm shifts in num_trials years.
    """

    return (num_successes + 1) / (num_trials + 2)

def laplace_rule_succ_NEW(success_years: List[int],
                          success_rate_func: Optional[Callable] = None,
                          year_to_predict: int = None) -> float:
    """
    Probability of seeing a new paradigm shift in the next year, given a list of
    when each paradigm shift occurred.

    success_years: list of years when paradigm shifts occurred
    success_rate_func: function that takes a year and returns the "success rate",
        which can be thought of as being proportional to the number of
        "effective researcher years" (default: constant function that returns 1)
    year_to_predict: year for which we want to predict the probability of a new
        paradigm shift (default: either current year + 1 or the year after the
        last paradigm shift, whichever is later)

    P(Success_t) = r(t) \\frac{\sum r(t_i)x_i + 1}{\sum r(t_i) + 2}
    """
    if success_rate_func is None:
        success_rate_func = lambda _: 1
    
    if year_to_predict is None:
        year_to_predict = max(current_year, max(success_years))+1

    succ_rate_sum_succ = sum([success_rate_func(yr) for yr in success_years])
    succ_rate_sum = sum([success_rate_func(yr)
                         for yr in range(min(success_years), year_to_predict)])
    
    return success_rate_func(year_to_predict) * (succ_rate_sum_succ + 1) / (succ_rate_sum + 2)

# predict number of paradigm shifts by year using branching
def branching_distributions(selected_years: List[int],
                            success_rate_func: Optional[Callable] = None,
                            num_yrs_forward: int = 30) -> List[List[float]]:
    """
    In any given year, how many paradigm shifts will we have seen by that year?

    Uses a branching algorithm - in each year, take last year's distribution over
    the number of paradigm shifts and for each possibility in that distribution,
    apply Laplace's rule of succession to figure out how likely we are to see a
    new paradigm shift in the next year.

    Eg. If the n-th year has a 90% chance of 0 paradigm shifts and a 10% chance of
    1 paradigm shift, we can figure out the distribution over the (n+1)-th year
    by using Laplace's rule of succession for both "branches" (ie the branch with
    a paradigm shift in year n and the branch without it), and then weigh
    the distribution in each branch by the likelihood of being in that branch.
    """
    num_shifts_probs = [[0.0 for _ in range(num_yrs_forward+1)]
                        for _ in range(num_yrs_forward)]
    for idx_year in range(num_yrs_forward):
        if idx_year == 0:
            # base case: only one branch so far, create 2 branches using Laplace's rule
            prob_shift = laplace_rule_succ_NEW(selected_years, success_rate_func)
            num_shifts_probs[0][1] = prob_shift
            num_shifts_probs[0][0] = 1 - prob_shift
            continue

        # for each branch, apply Laplace's rule of succession
        for branch, prob_branch in enumerate(num_shifts_probs[idx_year-1]):
            if prob_branch == 0:
                continue

            # ASSUMPTION: in order to keep the number of branches manageable (ie not exponential),
            # we merge all the branches that have seen the same number of paradigm shifts,
            # and assume that the paradigm shifts were evenly distributed in them.
            # Concretely, this means that given S paradigm shifts in T years, we assume
            # the shifts happened at forall i in [0, S]: (2i+1)T//(2S)
            branch_shifts = selected_years + [current_year+1 + (((2*i + 1) * idx_year) // (2*branch))
                                              for i in range(branch)]

            prob_shift = laplace_rule_succ(branch_shifts,
                                           success_rate_func,
                                           current_year + idx_year + 1)
            num_shifts_probs[idx_year][branch+1] += prob_branch * prob_shift
            num_shifts_probs[idx_year][branch] += prob_branch * (1 - prob_shift)
        
    return num_shifts_probs

#%%
# get the data
num_shifts_probs = branching_distributions(selected_years, num_yrs_forward)

#%%
# Create the plot using Plotly
fig = go.Figure()

for n_shifts in range(len(num_shifts_probs[0])):
    n_shifts_probs = [round(year_data[n_shifts], 3) for year_data in num_shifts_probs]

    fig.add_trace(go.Bar(
        x=[str(current_year+year+1) for year in range(len(num_shifts_probs))], 
        y=n_shifts_probs, 
        name=f"{n_shifts} shift{'s' if n_shifts != 1 else ''}",
        showlegend=n_shifts < 14 and max(n_shifts_probs) > 0.02,
    ))

fig.update_layout(
    title='Probability of seeing a given number of paradigm shifts by year',
    xaxis_title='Year',
    yaxis_title='Probability',
    barmode='stack'
)

st.plotly_chart(fig)

#%%
st.write("## Probability of seeing a given number of paradigm shifts by year")

num_shifts_from_user = st.slider('Number of paradigm shifts', 1, 10, 3)
prob_of_reaching_num_shifts = [sum(year_probs[num_shifts_from_user:]) for year_probs in num_shifts_probs]
fig = go.Figure()
fig.add_trace(go.Bar(
    x=[str(current_year+year+1) for year in range(len(num_shifts_probs))],
    y=[round(prob, 3) for prob in prob_of_reaching_num_shifts],
    name=f'{num_shifts_from_user} shifts',
))
fig.update_layout(
    title=f"Probability of seeing {num_shifts_from_user} "
            + f"paradigm shift{'s' if num_shifts_from_user != 0 else ''} by year",
    xaxis_title='Year',
    yaxis_title='Probability',
)
st.plotly_chart(fig)
# %%
# TODO check if you get the same results if you run it month-by-month instead of year-by-year
