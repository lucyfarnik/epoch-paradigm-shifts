#%%
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import plotly.express as px
from typing import List, Optional
import numpy as np
from scipy.optimize import curve_fit


#%%
def get_arxiv_papers_count(categories: List[str], year: int, month: Optional[int] = None):
    # Set the base URL for the ArXiv API
    base_url = 'http://export.arxiv.org/api/query?'
    
    # Set the search query parameters
    categories_query = ' OR '.join([f'cat:{category}' for category in categories])
    if month is None:
        query = f'({categories_query}) AND submittedDate:[{year}00000000 TO {year}12319999]'
    else:
        query = f'({categories_query}) AND submittedDate:[{year}{month:02d}01000000 TO {year}{month:02d}31999999]'
    params = {
        'search_query': query,  
        'max_results': 1,
    }
    
    # Make the request to the ArXiv API
    response = requests.get(base_url, params=params)
    
    # Parse the response
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        for child in root:
            if 'totalResults' in child.tag:
                return int(child.text)
    else:
        print(f'Error: Unable to fetch data. Status Code: {response.status_code}')
    return None

# Example Usage
categories = ['cs.AI', 'cs.LG', 'cs.NE', 'stat.ML']
year = 2022
month = 8  # August
papers_count = get_arxiv_papers_count(categories, year, month)
print(f'Number of Papers Published in {categories} in {year}-{month}: {papers_count}')

# %%
pub_counts = []
for year in range(1993, 2023):
    papers_count = get_arxiv_papers_count(categories, year)
    pub_counts.append({
        'year': year,
        'papers_count': papers_count,
    })

#%%
# make a line chart
df = pd.DataFrame(pub_counts)
fig = px.scatter(df, x='year', y='papers_count', title='Number of Papers Published in ArXiv')
fig.show()

df.to_csv('arxiv_papers_count.csv', index=False)

# %%
# fit an exponential curve
def exponential_func(x, a, b, c, d):
    return a * np.exp(b * (x+c)) + d

x = df['year'].values
y = df['papers_count'].values
popt, _ = curve_fit(exponential_func, x, y, p0=(1e-5, 1, -2000, 0))
print(popt)

fig = px.scatter(df, x='year', y='papers_count', title='Data and Fitted Exponential Curve')
fig.add_scatter(x=df['year'], y=exponential_func(df['year'], *popt), mode='lines', name='Fitted Curve')
fig.show()
# %%
# fit a logistic curve
def logistic_func(x, a, b, c):
    return a / (1 + np.exp(-b * (x-c)))

x = df['year'].values
y = df['papers_count'].values
popt, _ = curve_fit(logistic_func, x, y, p0=(1e5, 1, 2020))
print(popt)

x_pred = np.arange(1993, 2053)
fig = px.scatter(df, x='year', y='papers_count', title='Data and Fitted Logistic Curve')
fig.add_scatter(x=x_pred, y=logistic_func(x_pred, *popt), mode='lines', name='Fitted Curve')
fig.show()
# %%
