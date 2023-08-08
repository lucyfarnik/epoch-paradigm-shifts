#%%
"""
Insights: some of the top papers are indeed paradigm shifts (transformers, deep learning, CNNs).
Others are just important within a certain field of ML
(eg. "Distinctive Image Features from Scale-Invariant Keypoints") or a specific
application of ML (eg. "Multitask learning and benchmarking with clinical time series data").
There are also papers like Adam or A* Sampling which are important advancements but
not necessarily paradigm-shifting.

"""
import pandas as pd
import requests
import plotly.graph_objs as go

#%%
df = pd.read_csv('ml_papers_citations.csv')
df
# %%
print('Papers with the largest number of citations:')
df.sort_values('citations', ascending=False).head(30)
# %%
df.sort_values('citations', ascending=False).head(30)['title'].values

# %%
def plot_cites_by_year(title):
    res = requests.get(f'https://api.openalex.org/works?filter=title.search:"{title}"')
    citations = res.json()['results'][0]['counts_by_year']

    x = [d['year'] for d in citations]
    y = [d['cited_by_count'] for d in citations]

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_layout(title=title, yaxis_title='Citation Count')
    fig.show()

plot_cites_by_year('Attention Is All You Need')
# %%
plot_cites_by_year('Deep Residual Learning for Image Recognition')
# %%
#TODO
# try weighing the citations over time by the total number of papers in that given year (to account for the fact that the field has grown)
# as a first pass I could also weigh it by the year the paper itself was published
# try to create a continuous bar rather than trying to do binary classification