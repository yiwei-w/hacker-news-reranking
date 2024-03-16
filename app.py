import requests
import streamlit as st
from sentence_transformers import CrossEncoder
import concurrent.futures
import json
import os

CACHE_FILE = "hn_stories_cache.jsonl"

def fetch_story_details(story_id):
    story_details_url = 'https://hacker-news.firebaseio.com/v0/item/{}.json'
    story_response = requests.get(story_details_url.format(story_id))
    story_data = story_response.json()
    text = story_data.get('text', '')
    title = story_data.get('title', 'No Title')
    hn_url = f"https://news.ycombinator.com/item?id={story_id}"
    if text:
        title_text = title + "\n" + text
    else:
        title_text = title
    return {'title': title, 'hn_url': hn_url, "text": text, "title_text": title_text}

def fetch_top_hn_stories(limit=500):
    top_stories_url = 'https://hacker-news.firebaseio.com/v0/topstories.json'
    response = requests.get(top_stories_url)
    story_ids = response.json()[:limit]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_story_details, story_id) for story_id in story_ids]
        top_stories = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Save the fetched stories to a local jsonlines file
    with open(CACHE_FILE, "w") as f:
        for story in top_stories:
            f.write(json.dumps(story) + "\n")

    return top_stories

def load_cached_stories():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return [json.loads(line) for line in f]
    return None

def rerank_stories(stories, user_description):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [(user_description, story['title_text']) for story in stories]
    scores = model.predict(pairs)
    return [story for _, story in sorted(zip(scores, stories), key=lambda x: x[0], reverse=True)]


st.title('Hacker News Top Stories Reranker')

# Display a loading spinner while fetching stories
with st.spinner('Fetching top stories...'):
    # Load cached stories if available, otherwise fetch from HN API
    top_stories = load_cached_stories()
    if top_stories is None:
        top_stories = fetch_top_hn_stories()

# Input for personal interests
user_description = st.text_area("Enter your interests:", "")

col1, col2 = st.columns(2)
with col1:
    submit_clicked = st.button('Rerank Stories')
with col2:
    refresh_clicked = st.button('Refresh Cache')

# Separate the buttons from the display logic for stories
if submit_clicked:
    # Rerank the stories based on the user's interests
    top_stories = rerank_stories(top_stories, user_description)

    # Display the (re)ranked stories in a single column
    for i, story in enumerate(top_stories):
        st.write(f"{i+1}. [{story['title']}]({story['hn_url']})")

if refresh_clicked:
    with st.spinner('Refreshing cache...'):
        top_stories = fetch_top_hn_stories()
    st.write("Cache refreshed successfully!")
    top_stories = rerank_stories(top_stories, user_description)

    # Display the (re)ranked stories in a single column
    for i, story in enumerate(top_stories):
        st.write(f"{i+1}. [{story['title']}]({story['hn_url']})")

