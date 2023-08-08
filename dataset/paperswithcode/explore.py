#%%
import os
import requests
import pandas as pd
import re

#%%
assert os.path.exists('papers-with-abstracts.json'), 'Please download the dataset first: https://production-media.paperswithcode.com/about/papers-with-abstracts.json.gz'
df = pd.read_json('papers-with-abstracts.json')
df
# %%
# see what percentage of abstracts contain the word 'learn'
print('Percentage of abstracts containing the word "learn":',
      df['abstract'].str.contains('learn').mean())

# %%
# same but for different keywords
keywords = ['learn', 'deep', 'neural network', 'transformer', 'NLP',
            'natural language processing', 'machine learning', 'ML',
            'computer vision', 'CV', 'reinforcement learning', 'RL', 'supervised']
# add machine learning libraries to the keywords (from ml_libs.txt)
with open('ml_libs.txt') as f:
    keywords += f.read().splitlines()

for keyword in keywords:
    print(f'{keyword:>30}: {df["abstract"].str.contains(keyword).mean():.3f}')

# %%
# how many papers contain at least one of these?
print('At least one of these keywords:', df['abstract'].str.contains('|'.join(keywords)).mean())
# %%
# of the ones that contain at least one of these, print a few random ones
print('Random abstracts containing at least one of these keywords:')
df[df['abstract'].fillna('').str.contains('|'.join(keywords))].sample(10)['abstract'].tolist()
# %%
# how many papers contain at least two of the keywords?

# Create a function that counts unique keyword matches in a text
pattern = '|'.join(keywords)
def count_unique_matches(text):
    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text)
    # Return the number of unique matches
    return len(set(matches))

df['match_counts'] = df['abstract'].fillna('').apply(count_unique_matches)
print('At least two of these keywords:', (df['match_counts'] >= 2).mean())
# %%
# print a few abstracts that contain at least 2 keywords
print('Random abstracts containing at least 2 keywords:')
df[df['match_counts'] >= 2].sample(10)['abstract'].tolist()
# %%
# print a few abstracts that don't contain the first 3 keywords
print('Random abstracts not containing the first 3 keywords:')
df[~df['abstract'].fillna('').str.contains('|'.join(keywords[:3]))].sample(10)['abstract'].tolist()




# %%
# print percentage of papers that were written by one of the authors in ml_authors.txt
with open('ml_authors.txt') as f:
    ml_authors = f.read().splitlines()
df['ml_author'] = df['authors'].apply(lambda x: any(author in x for author in ml_authors))
print('Percentage of papers written by one of the authors in ml_authors.txt:',
      df['ml_author'].mean())
# %%
# print percentage of papers that were published in one of the conferences in ml_conferences.txt
with open('ml_conferences.txt') as f:
    ml_conferences = f.read().splitlines()
df['ml_conference'] = df['proceeding'].fillna('').apply(lambda x: any(conf in x
                                                                      for conf in ml_conferences))
print('Percentage of papers published in one of the conferences in ml_conferences.txt:',
        df['ml_conference'].mean())
# %%
# what percentage of papers match at least one of these 3 criteria?
df['one_criterion'] = (df['ml_author'] | df['ml_conference'] | (df['match_counts'] >= 2))
print('At least one of these 3 criteria:', df['one_criterion'].mean())


# %%
# save the list of papers to a file (only save arxiv_id, title, and authors)
# only save papers that match at least one of the 3 criteria
# convert authors into a string
df['authors'] = df['authors'].apply(lambda x: ', '.join(x))
df[df['one_criterion']][['arxiv_id', 'title', 'authors']].to_csv('ml_papers.csv', index=False)
# %%

