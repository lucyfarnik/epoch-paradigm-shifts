# Paradigm shifts in ML
The purpose of this project is to identify a list of paradigm shifts in the history
of machine learning, and use this to create a predictive model of when we are
likely to see the next ML paradigm shift.

This code is currently a bit of a mess — I was rapidly prototyping and not focusing
on keeping things neat and tidy. It's all a work in progress right now.
If anything isn't clear, please ask.

**IMPORTANT NOTE: the files with the actual data are currently not included in this repo**

## Dataset
We start with the PapersWithCode database, which is in `dataset/paperswithcode/papers-with-abstracts.json`.
This file contains tons of papers, some are about ML and others aren't.

### Figuring out which papers are about ML
We have a list of ML conferences in `dataset/paperswithcode/ml_conferences.txt`, and a list
of the libraries commonly used in ML papers is in `dataset/paperswithcode/ml_libs.txt`.

We also use the `scholarly` API (ie. Google Scholar) to get a list of researchers
working on ML (based on their research interests) — the code for this is in
`dataset/ml_authors.py` and the results are in `dataset/paperswithcode/ml_authors.txt`.

We then combine all of this along with looking for certain keywords in the abstract
to classify papers as ML or not-ML. This is done in the last part of `dataset/paperswithcode/explore.py`,
the results (ie. all ML papers) are stored in `dataset/paperswithcode/ml_papers.csv`.

### Finding citations
There are currently major problems with this part, we're working to resolve them.

Currently, it works by pinging the OpenAlex API with the title of the paper, but
sometimes (eg. for the paper titled "A* sampling") this doesn't work as intended.

This is implemented in `dataset/get_citations.py`, the results are `dataset/ml_papers_citations.csv`.

### Exploring the dataset and a web app interface
There's some exploratory code in `dataset/explore_dataset.py`.

I also build a small web app to make this easier to play around with, that's in
`dataset/webapp.py`. To run it, install streamlit and then run `streamlit run dataset/explore_dataset.py`.

## Predictive model
The web app in `predictive_webapp.py` lets users enter in their list of paradigm
shifts and use this to generate a probability distribution over the coming years
to predict when there is likely to be the next paradigm shift.
Run it with `streamlit run predictive_webapp.py`.
