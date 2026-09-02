"""
Sermon lookup.

Three kinds of question reach the bot, and only one of them is a similarity
problem:

  "Jonathan Edwards"          -> who preached it   (exact, from the catalog)
  "what is stewardship"       -> a site category   (exact, from the catalog)
  "how do I forgive someone"  -> what it is about  (vector search)

Answering the first two with a vector search is what made them fail: an author's
name is a few words buried in a 500-word chunk, and a category is not in the
sermon text at all. So each is routed to the index that can actually answer it,
and only genuine topic questions go to Pinecone.
"""

import logging
import re

from app.catalog import content_words, tokenize

logger = logging.getLogger(__name__)

TOP_K = 25

# A chunk has to clear both bars: a decent blended score, and enough raw
# semantic similarity that a lucky keyword hit alone cannot carry it.
MIN_HYBRID_SCORE = 0.42
MIN_SEMANTIC_SCORE = 0.33

_SEMANTIC_WEIGHT = 0.6
_TEXT_WEIGHT = 0.4

# A title match says what a sermon is *about*; a body match only says the word
# came up. "children" in "DISCIPLINE Your Children" is the sermon, "children" in
# "The Holy Spirit's Intercession" is "children of God" in one passing sentence.
# This rides on top of the blend rather than replacing part of it, so adding it
# can only promote a sermon, never cost one its place.
_TITLE_BONUS = 0.25

# The keyword fallback runs only when the vector search is empty, so it has to
# refuse rather than reach: "best pizza recipe" must stay unanswered even though
# some sermon somewhere contains the word "best".
MIN_FALLBACK_SCORE = 0.9

_URL_RE = re.compile(r"\((https?://[^)\s]+)\)")


def previously_shown(history: list) -> set:
    """Sermon links already sent in this conversation, so "more" means more."""
    return {
        url
        for message in history
        if message.get("role") == "assistant"
        for url in _URL_RE.findall(message.get("content", ""))
    }


def _overlap(words: set, text: str) -> float:
    if not words:
        return 0.0
    return len(words & set(tokenize(text))) / len(words)


# -------------------- Author and category lookups --------------------

def _rank_within(sermons: list, query: str, name_words: set) -> list:
    """
    Order a fixed set of sermons by whatever else the query asked for.

    "Tim Keller on anger" should lead with the anger sermon; a bare "Tim Keller"
    has nothing to sort by, so the catalog's own order stands.
    """
    extra = content_words(query) - name_words
    if not extra:
        return sermons

    scored = [
        (
            len(extra & s["_body_words"]) / len(extra) * _TEXT_WEIGHT
            + _overlap(extra, s["title"]) * _TITLE_BONUS,
            i,
            s,
        )
        for i, s in enumerate(sermons)
    ]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [s for _, _, s in scored]


def search_by_author(catalog, query: str, author_keys: list) -> tuple[list, str]:
    sermons, seen = [], set()
    for key in author_keys:
        for sermon in catalog.by_author.get(key, []):
            if sermon["url"] not in seen:
                seen.add(sermon["url"])
                sermons.append(sermon)

    name_words = {w for key in author_keys for w in tokenize(key)}
    names = [catalog.author_display[k] for k in author_keys if k in catalog.author_display]

    if len(names) == 1:
        label = names[0]
    elif len(names) == 2:
        label = f"{names[0]} and {names[1]}"
    else:
        label = ", ".join(names[:-1]) + f", and {names[-1]}"

    return _rank_within(sermons, query, name_words), label


def search_by_category(catalog, index, embed, query: str, category_key: str) -> tuple[list, str]:
    sermons = catalog.by_category.get(category_key, [])
    display = catalog.category_display(category_key)

    # Someone typing a bare category name gives nothing to sort by, so order the
    # category by how central each sermon is to it rather than by scrape order.
    ranked = _rank_by_similarity(index, embed, query, display, sermons)

    return _rank_within(ranked, query, set(tokenize(category_key))), display


def _rank_by_similarity(index, embed, query: str, category: str, sermons: list) -> list:
    """Order one category's sermons by similarity, keeping unmatched ones last."""
    if not sermons:
        return sermons

    try:
        results = index.query(
            vector=embed(query),
            top_k=TOP_K,
            include_metadata=True,
            filter={"category": category},
        )
    except Exception as e:
        logger.warning(f"Category-filtered query failed for {category!r}: {e}")
        return sermons

    best: dict[str, float] = {}
    for match in results.get("matches", []):
        url = (match.get("metadata", {}).get("url") or "").strip()
        if url:
            best[url] = max(best.get(url, 0), match.get("score", 0))

    if not best:
        return sermons

    order = {s["url"]: i for i, s in enumerate(sermons)}
    return sorted(
        sermons,
        key=lambda s: (-best.get(s["url"], 0), order[s["url"]]),
    )


# -------------------- Topic search --------------------

def score_matches(matches: list, query: str) -> list:
    """Blend Pinecone's similarity with keyword overlap on the title and body."""
    keywords = content_words(query)
    logger.info(f"Query keywords: {sorted(keywords)}")

    scored = []
    for match in matches:
        metadata = match.get("metadata", {})
        semantic = match.get("score", 0)

        title_score = _overlap(keywords, metadata.get("title", ""))
        text_score = _overlap(keywords, metadata.get("text", ""))

        match["keyword_score"] = text_score
        match["hybrid_score"] = (
            semantic * _SEMANTIC_WEIGHT
            + text_score * _TEXT_WEIGHT
            + title_score * _TITLE_BONUS
        )
        scored.append(match)

    scored.sort(key=lambda m: m["hybrid_score"], reverse=True)
    return scored


def search_by_topic(catalog, index, embed, query: str) -> list:
    """Vector search, falling back to a local keyword scan when it finds nothing."""
    results = index.query(vector=embed(query), top_k=TOP_K, include_metadata=True)
    scored = score_matches(results.get("matches", []), query)

    if scored:
        top = scored[0]
        logger.info(
            f"Top match: {top['metadata'].get('title')!r} "
            f"hybrid={top['hybrid_score']:.3f} semantic={top.get('score', 0):.3f}"
        )

    sermons, seen = [], set()
    for match in scored:
        metadata = match.get("metadata", {})
        if metadata.get("type", "sermon") == "bible":
            continue
        if match["hybrid_score"] <= MIN_HYBRID_SCORE:
            continue
        if match.get("score", 0) <= MIN_SEMANTIC_SCORE:
            continue

        url = (metadata.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)

        # Prefer the catalog entry: it carries the cleaned author list, and the
        # index stores a title per chunk that can be missing or truncated.
        sermon = catalog.by_url.get(url)
        sermons.append(sermon or {
            "title": (metadata.get("title") or "Sermon").replace('"', "").strip(),
            "url": url,
            "authors": [],
        })

    if sermons:
        return sermons

    # Nothing cleared the bar. Rather than telling someone we have no sermons on
    # children when three of them have "Children" in the title, fall back to a
    # plain word match over the catalog.
    logger.info("Vector search returned nothing above threshold; using keyword fallback")
    return catalog.keyword_search(query, min_score=MIN_FALLBACK_SCORE)


# -------------------- Entry point --------------------

def find_sermons(catalog, index, embed, query: str, exclude: set, limit: int) -> dict:
    """
    Resolve a query to sermons.

    Returns the page of results to show, the total available for this query, and
    a label describing what was searched, so the caller can say "3 of 14 by
    Jonathan Edwards" instead of silently truncating.
    """
    author_keys = catalog.find_authors(query)
    if author_keys:
        sermons, label = search_by_author(catalog, query, author_keys)
        kind = "author"
    else:
        category_key = catalog.find_category(query)
        if category_key:
            sermons, label = search_by_category(catalog, index, embed, query, category_key)
            kind = "category"
        else:
            sermons, label, kind = search_by_topic(catalog, index, embed, query), "", "topic"

    logger.info(f"Search kind={kind} label={label!r} candidates={len(sermons)}")

    remaining = [s for s in sermons if s["url"] not in exclude]
    return {
        "kind": kind,
        "label": label,
        "results": remaining[:limit],
        "total": len(sermons),
        "remaining": len(remaining),
        "exhausted": bool(sermons) and not remaining,
    }
