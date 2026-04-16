import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_most_similar_source(response: str, sermon_data: dict) -> tuple[str, str, str]:
    """
    Given an LLM response and the sermon_data dict, return:
      (best_match_url, random_url, random_category)

    NOTE: sermon_data is passed in as a dict, NOT a file path.
    This avoids the hardcoded path problem from the old repo.
    """
    contents = [s["content"] for s in sermon_data.values()]
    urls = [s["url"] for s in sermon_data.values()]
    url_category_pairs = [(s["url"], s["category"]) for s in sermon_data.values()]

    random_url, random_category = random.choice(url_category_pairs)

    all_texts = contents + [response]
    vectorizer = TfidfVectorizer().fit_transform(all_texts)
    cosine_similarities = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()
    most_similar_index = np.argmax(cosine_similarities)

    return urls[most_similar_index], random_url, random_category


def format_answer_with_sources(answer: str, source_url: str, random_url: str, random_category: str) -> str:
    """Append source links in the old bot's style."""
    source_phrases = [
        f"For more information on this topic, please refer to: {source_url}",
        f"This answer draws from the following source: {source_url}",
        f"The insights provided are based on content available at: {source_url}",
        f"For a deeper dive into this matter, consider visiting: {source_url}",
        f"The full sermon that inspired this answer is available at: {source_url}",
    ]
    second_phrases = [
        f"If you're interested in exploring the topic of {random_category}, you might find this resource enlightening: {random_url}",
        f"To broaden your understanding of {random_category}, consider reading: {random_url}",
        f"Expand your knowledge on {random_category} by visiting: {random_url}",
        f"For a related discussion on {random_category}, we recommend: {random_url}",
    ]
    return f"{answer}\n\n{random.choice(source_phrases)}\n\n{random.choice(second_phrases)}"