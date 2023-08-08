# Use the scholarly library to scrape Google Scholar for deep learning papers
#%%
import os
from scholarly import scholarly
import pandas as pd

#%%
#TODO: Use PapersWithCode to get list of papers, then use Scholarly to get citation counts
#? https://paperswithcode.com/api/v1/docs/
# TODO look at the abstract and look for keywords to make sure it's a DL paper; also consider classifiers (BERT etc)
# Search for deep learning papers 
query = scholarly.search_keyword('deep learning')

if os.path.exists('authors.txt'):
    with open('authors.txt', 'r') as f:
        processed_authors = [line.strip() for line in f.readlines()]
else:
    processed_authors = []

if os.path.exists('papers.csv'):
    papers = pd.read_csv('papers.csv')
else:
    papers = pd.DataFrame(columns=['title', 'citation', 'cites_id', 'pub_year', 'num_citations', 'author', 'author_id'])

# Iterate through authors
for author in query:
    # Skip if author has already been processed
    if author['name'] in processed_authors:
        continue

    print(author['name'])

    # get the author's papers
    author = scholarly.fill(author)

    # loop through the papers, append to papers dataframe
    titles = []
    citations = []
    cites_ids = []
    pub_years = []
    num_citations = []
    for paper in author['publications']:
        # get publication year
        if 'pub_year' in paper['bib']:
            pub_year = paper['bib']['pub_year']
        elif paper['bib']['title'][-4:].isdigit(): # sometimes the year is in the title
            pub_year = paper['bib']['title'][-4:]
        else:
            pub_year = None
            # print('No publication year found', paper)

        # get the ID of the paper
        if 'cites_id' in paper and len(paper['cites_id']) > 0:
            cites_id = ','.join(paper['cites_id'])
        else:
            cites_id = None
        titles.append(paper['bib']['title'])
        citations.append(paper['bib']['citation'])
        cites_ids.append(cites_id)
        pub_years.append(pub_year)
        num_citations.append(paper['num_citations'])
    papers = pd.concat([papers, pd.DataFrame({
        'title': titles,
        'citation': citations,
        'cites_id': cites_ids,
        'pub_year': pub_years,
        'num_citations': num_citations,
        'author': author['name'],
        'author_id': author['scholar_id']
    })], ignore_index=True)
        

        # print(paper['bib']['title'], pub_year, paper['num_citations'])
    print('Currently have', len(papers), 'papers')
    # save to csv
    papers.to_csv('papers.csv')
    # add the name of the author to authors.txt
    with open('authors.txt', 'a') as f:
        f.write(author['name'] + '\n')
# %%
