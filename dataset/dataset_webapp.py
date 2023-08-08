#%%
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
import urllib.parse

# %%
df = pd.read_csv('ml_papers_citations.csv')
# sort by citation count
df = df.sort_values('citations', ascending=False)

#%%
def get_citations_over_time(title: str) -> pd.DataFrame:
    res = requests.get(f'https://api.openalex.org/works?filter=title.search:"{title}"')
    citations = res.json()['results'][0]['counts_by_year']

    x = [d['year'] for d in citations]
    y = [d['cited_by_count'] for d in citations]

    return pd.DataFrame({'year': x, 'citations': y})

#%%
# Function to display the top N papers
def display_papers(n, start=0):

    sorted_papers = df.sort_values('citations', ascending=False)
    selected_papers = sorted_papers[start:start + n]
    for i, row in selected_papers.iterrows():
        paper_title = row['title']
        citations = row['citations']
        arxiv_id = row['arxiv_id']
        col1, col2, col3 = st.columns([6, 1, 1])
        with col1:
            st.markdown(f"**{paper_title}** - Citations: {citations}")
        with col2:
            st.markdown(f"[Paper](https://scholar.google.com/scholar?q={urllib.parse.quote(paper_title)})")
        with col3:
            if not pd.isna(arxiv_id):
                st.markdown(f"[PDF](https://arxiv.org/pdf/{arxiv_id}.pdf)")

        if st.button("Citations over time", key=f'cite-dist-{arxiv_id if not pd.isna(arxiv_id) else i}'):
            citation_data = get_citations_over_time(paper_title)
            fig = px.bar(citation_data, x='year', y='citations',
                         title=f'Citations Over Time for "{paper_title}"')
            fig.update_layout(showlegend=False, template="plotly_white")
            st.plotly_chart(fig)

        st.write("---")

#%%
# Streamlit app
st.title("ML Papers Visualization")
N = st.slider("Select number of papers to display per page:", 5, 100, 30)

# Pagination control
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 0

max_pages = (len(df) - 1) // N
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    if st.button("Previous Page") and st.session_state['current_page'] > 0:
        st.session_state['current_page'] -= 1
with col2:
    st.markdown(
        f"<div style='text-align: center;'>Page {st.session_state['current_page'] + 1}/{max_pages + 1}</div>",
        unsafe_allow_html=True)
with col3:
    if st.button("Next Page") and st.session_state['current_page'] < max_pages:
        st.session_state['current_page'] += 1

start = st.session_state['current_page'] * N

st.markdown("---")

display_papers(N, start)
#%%
#TODO
# try weighing the citations over time by the total number of papers in that given year (to account for the fact that the field has grown)
# as a first pass I could also weigh it by the year the paper itself was published
# try to create a continuous bar rather than trying to do binary classification