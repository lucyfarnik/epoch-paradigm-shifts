#%%
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import datetime
from typing import List, Optional, Callable

#%%
st.write("""
         # Predictive model of future paradigm shifts
         This app predicts when the next ML paradigm shifts will occur given your
         beliefs about which past innovations count as paradigm shifts, as well as
         the future of the ML field.
         
         Developed by [Lucy Farnik](https://www.linkedin.com/in/lucy-farnik/),
         [Francesca Sheeka](https://www.linkedin.com/in/fsheeka/),
         and [Pablo Villalobos](https://www.linkedin.com/in/pablo-villalobos-sanchez/).
         Please send any questions, bug reports, or feature requests to
         [lucyfarnik@gmail.com](mailto:lucyfarnik@gmail.com).
         """)

#%%
current_year = datetime.datetime.now().year

#%%
st.write("""
         ## Parameter 1: Select paradigm shifts
         Which innovations would you consider to be a paradigm shift?
         You can select from the list below, or add your own by typing it in the
         text box (make sure to include the year in brackets).
         """)
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
        # 'Foundation models (2020)'
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
            if year > current_year or year < 1000:
                raise ValueError
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
st.write("""
         ## Parameter 2: How far into the future do you want to predict?
         This effectively just sets the zoom level of the charts.
         """)
num_yrs_forward = st.slider('Number of years to predict', 10, 100, 30, 5)
#%%
st.write("""
        ## Parameter 3: How much effort do you expect to go into ML research in the future?
        It's easy to see that the ML field is growing — over the last 3 decades,
         the number of arXiv papers published in ML has grown exponentially.
         Obviously exponential growth cannot last forever. Where do you expect it
         to plateau? What number of ML papers per year do you think will be the maximum?

         For reference, the chart includes the number of ML papers submitted to 
         arXiv each year from 1993 to 2021. **Important note**: back in the 90s and
         before, not a lot of people were using arXiv, so the number of papers
         submitted back then should not inform your views too much.
         
         This is why we also provide a parameter to set the minimum number of
         papers per year in the past — think of it as the number of ML papers
         published at the time of the first paradigm shift.

         (For the math nerds: you are actually choosing the supremum and infinimum
         of the logistic function. Once you pick these, the app fits the logistic
         function onto the data from 2010 onward)
         """)

papers_count_df = pd.read_csv('arxiv_papers_count.csv')

detailed_logistic_control = st.checkbox('I want more detailed control over this function')

if detailed_logistic_control:
    supremum = st.number_input('Supremum', value=int(1e5))
    infinimum = st.number_input('Infinimum', value=500)
    logistic_growth = st.number_input('Growth', value=0.35)
    logistic_midpoint = st.number_input('Midpoint', value=2023)
else:
    supremum = st.select_slider('Maximum number of papers per year',
                                options=[4e4, 5e4, 6e4, 7e4, 8e4, 9e4,
                                        1e5, 3e5, 5e5, 7e5, 1e6, 5e6, 1e7, 5e7],
                                value=1e5, format_func=lambda x: f'{x:,.0f}')
    infinimum = st.select_slider('Minimum number of papers per year',
                                options=[10, 30, 50, 70, 100, 150, 200, 300, 400, 500, 700,
                                        1000, 1500, 2000, 3000, 4000, 5000, 7000, 10000],
                                value=500, format_func=lambda x: f'{x:,.0f}')

    def logistic_func_fixed_sup(x, growth, midpoint):
        return (supremum - infinimum) / (1 + np.exp(-growth * (x-midpoint))) + infinimum

    papers_2010_onward = papers_count_df[papers_count_df['year'] >= 2010]
    (logistic_growth, logistic_midpoint), _ = curve_fit(logistic_func_fixed_sup,
                                                        papers_2010_onward['year'].values,
                                                        papers_2010_onward['papers_count'].values,
                                                        p0=(1, 2020))
    
def logistic_func(x):
    return (supremum - infinimum) / (1 + np.exp(-logistic_growth * (x-logistic_midpoint))) + infinimum


x_pred = np.arange(1993, current_year+num_yrs_forward)
fig = go.Figure()
fig.add_trace(go.Scatter(x=papers_count_df['year'], y=papers_count_df['papers_count'],
                         mode='markers', name='arXiv Papers'))
fig.add_trace(go.Scatter(x=x_pred, y=logistic_func(x_pred),
                         mode='lines', name='Fitted Curve'))

fig.update_layout(title='Data and Fitted Logistic Curve',
                  xaxis_title='Year', yaxis_title='Number of papers')

st.plotly_chart(fig)

#%%
st.write("""
         ## Parameter 4: Are ideas getting harder to find? If so, by how much?
         Some people believe that the "low-hanging fruit" of ML has already been
         picked, and that future innovations will be harder to find. If you believe
         this, you can use this parameter to model this. If you don't believe this,
         you can set this parameter to 0.

         (Sidenote: there is a [paper](https://web.stanford.edu/~chadj/IdeaPF.pdf)
         on this phenomenon in science in general.)
         """)
innov_decl_perc = st.slider('In each year, how much harder is it to find new ideas?',
                            min_value=0., max_value=10., value=1.5, step=0.1, format='%.1f%%')

st.write("""
         Here's how that interacts with the growth of ML set in the previous parameter:
         """)
def ml_growth_func(x):
    growth_term = logistic_func(x)
    innov_decl_term = (1 - innov_decl_perc/100)**(x-2000)
    return growth_term * innov_decl_term

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_pred, y=ml_growth_func(x_pred), mode='lines', name='ML Growth'))
fig.add_trace(go.Scatter(x=x_pred, y=logistic_func(x_pred),
                         mode='lines', name='Original Curve', line=dict(dash='dash')))

fig.update_layout(title='Growth of ML with Innovation Decline',
                    xaxis_title='Year', yaxis_title='Number of importance-adjusted papers')

st.plotly_chart(fig)

# %%
st.write("## Results: Probability of seeing a given number of paradigm shifts by year")

def laplace_rule_succ(success_years: List[int],
                      success_rate_func: Optional[Callable[[int], float]] = None,
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


    result = success_rate_func(year_to_predict) * (succ_rate_sum_succ + 1) / (succ_rate_sum + 2)

    if result < 0 or result > 1:
        raise ValueError(f'Probability of seeing a new paradigm shift in year {year_to_predict} '
                         + f'is {result}, which is outside the range [0, 1].')

    return result.item()

# predict number of paradigm shifts by year using branching
def branching_distributions(selected_years: List[int],
                            success_rate_func: Optional[Callable[[int], float]] = None,
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
            prob_shift = laplace_rule_succ(selected_years, success_rate_func)
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
ml_growth_func_max = max([ml_growth_func(y) for y in range(1900,
                                                           current_year+num_yrs_forward)])
success_rate_func = lambda year: ml_growth_func(year) / ml_growth_func_max
num_shifts_probs = branching_distributions(selected_years, success_rate_func, num_yrs_forward)

#%%
# Create the plot using Plotly
fig = go.Figure()

for n_shifts in range(len(num_shifts_probs[0])):
    n_shifts_probs = [round(year_data[n_shifts], 3) for year_data in num_shifts_probs]

    fig.add_trace(go.Bar(
        x=[str(current_year+year+1) for year in range(len(num_shifts_probs))], 
        y=n_shifts_probs, 
        name=f"{n_shifts} shift{'s' if n_shifts != 1 else ''}",
        showlegend=(n_shifts < 14 and max(n_shifts_probs) > 0.02),
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
st.write("""
         ## Appendix: How the model works
         ### Modified Laplace's rule of succession
         The core of the model is Laplace's rule of succession, modified to account
         for a non-constant probability of success:
         $$P(Success_t) = r(t) \\frac{\sum r(t_i)x_i + 1}{\sum r(t_i) + 2}$$.

         $r(t)$ is the rate of success in year $t$, which in our case is set to
         the logistic function defined above, multiplied by $(1 - d)^{(t-2000)}$,
         where $d$ is the rate at which ideas are getting harder to find.
         It's visualized in the chart in the "ideas getting harder to find" section.
         To make this a valid probability, we normalize r(t) by dividing it by
         its maximum value.

         $x_i$ is a binary function indicating whether there was a success (ie.
         a paradigm shift) in year $i$.

         ### Branching algorithm
         We then use a branching algorithm to predict the number of paradigm shifts
         in each year. It works as follows:
         - In the first year, there is only one branch, so we apply Laplace's rule
         to figure out the probability of seeing a paradigm shift in that year.
         This gives us two "branches" — one with a paradigm shift in the first year,
         and one without (you can think of these as different timelines, Everett
         branches, etc.). Laplace's rule gives us the probability of each branch
         becoming reality.
         - Then for year $t$, take the branches from year $t-1$, and for each branch,
         use Laplace's rule to determine the probability of seeing a paradigm shift
         in year $t$ in this branch. Once this is complete for all branches, merge
         the branches with the same number of paradigm shifts.

         Note that there are a few simplifying assumptions here, as well as a few
         imprecisions in the model that we had to introduce to make it computationally
         tractable. First, we assume that there can never be more than 1 paradigm
         shift in a year. Second, note that we are merging all the branches that
         have had the same number of paradigm shifts (this takes the computational
         complexity of the algorithm down from exponential to quadratic).
         This means that we do not distinguish between a branch that saw its only
         paradigm shift in 2025 and a branch that had its only paradigm shift in 2030.
         This would not make a difference if $r(t)$ was constant, but since it's not,
         it makes our slightly imprecise. However, note that our modified Laplace's
         rule does actually need to know when the paradigm shifts occurred,
         so we assume that they were evenly spread out throughout the branch,
         using the formula `[current_year+1 + (((2*i + 1) * idx_year) // (2*n_shifts_in_branch))
         for i in range(n_shifts_in_branch)]`.
         """)