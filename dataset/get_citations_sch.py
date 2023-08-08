#%%
# load the paperswithcode/ml_papers dataset and get the citations for each of them
# using the arxiv API
import os
from scholarly import scholarly
import requests
import pandas as pd
from tqdm import tqdm

#%%
df = pd.read_csv('paperswithcode/ml_papers.csv')

#%%
# get the citation count for each paper and put them into df['citations'] (with tqdm)
df['citations'] = 0
for i in tqdm(range(len(df))):
    search_query = scholarly.search_pubs(df.loc[i, 'title'])#+ ' ' + row['authors'])
    try:
        publication = next(search_query)
        df.loc[i, 'citations'] = publication.citedby
    except StopIteration:
        print("Publication not found on Google Scholar.")
        df.loc[i, 'citations'] = None


# %%
