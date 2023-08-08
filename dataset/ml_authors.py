#%%
from scholarly import scholarly

#%%
# get the names of all authors who are interested in ML
keywords = ['machine learning', 'deep learning', 'natural language processing',
            'computer vision', 'reinforcement learning']
authors = set()
for i, keyword in enumerate(keywords):
    search_query = scholarly.search_keyword(keyword)
    keyword_count = 0
    for author in search_query:
        authors.add(author['name'])
        keyword_count += 1
        if keyword_count >= 500:
            break

#%%
# save the list of authors to a file
with open('ml_authors.txt', 'w') as f:
    for author in authors:
        f.write(author + '\n')


# %%
