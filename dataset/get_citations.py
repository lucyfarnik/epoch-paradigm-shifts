#%%
# load the paperswithcode/ml_papers dataset and get the citations for each of them
# using the arxiv API
import os
import requests
import pandas as pd
from tqdm import tqdm

#%%
while True:
    try:
        if os.path.exists('ml_papers_citations.csv'):
            df = pd.read_csv('ml_papers_citations.csv')
        else:
            assert os.path.exists('paperswithcode/ml_papers.csv'), 'Get the dataset by running paperswithcode/explore.py'
            df = pd.read_csv('paperswithcode/ml_papers.csv')
            df['citations'] = None
        # df

        if os.path.exists('ml_papers_citations_index.txt'):
            with open('ml_papers_citations_index.txt') as f:
                start_index = int(f.read())
        else:
            start_index = 0
        # f'{start_index=}'

        for i in tqdm(range(start_index, len(df))):
            # save every 100 iterations
            if i % 100 == 0:
                df.to_csv('ml_papers_citations.csv', index=False)
                # save index
                with open('ml_papers_citations_index.txt', 'w') as f:
                    f.write(str(i))

            # get data from OpenAlex
            res = requests.get(f'https://api.openalex.org/works?filter=title.search:"{df.loc[i, "title"]}"')
            # if no results or no citation count, skip
            if not 'results' in res.json() or len(res.json()['results']) == 0:
                print(f'No results for {df.loc[i, "title"]}')
                continue
            if 'cited_by_count' not in res.json()['results'][0]:
                print(f'No citation count for {df.loc[i, "title"]}')
                continue
            
            # save the citation count
            df.loc[i, 'citations'] = res.json()['results'][0]['cited_by_count']
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(e)
        continue
# %%
