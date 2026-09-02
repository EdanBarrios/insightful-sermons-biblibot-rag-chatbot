"""
In-memory catalog of the sermon corpus, built from data/sermon_data.json.

Pinecone handles semantic similarity, but two of the ways people actually search
are not semantic at all:

  * by author  ("Jonathan Edwards")
  * by category ("stewardship" — a section that exists on the website)

Both are exact-match lookups over facts that live in the sermon text and the
scraped metadata, so they are answered from this catalog rather than from a
vector search that can only ever approximate them.
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).parent.parent / "data" / "sermon_data.json"

# -------------------- Author extraction --------------------

# A single name token: "Tim", "C.S.", "Hung-Biu".
_TOKEN = r"[A-Z][A-Za-z'’.\-]*"
# Middle initials and surname particles, which may sit between forename and
# surname: "Richard N. Bolles", "Leonard J. Vander Zee".
_MID = r"(?:[A-Z]\.?|Van|Vander|Von|De|Del|Da|La|Le|Mc|Mac|St\.?)"
# A person is forename + surname, and nothing more. Attributions run straight
# into the body text ("A Lesson from Tim Keller If Christ is who He said..."),
# so a pattern that keeps consuming capitalised words invents authors.
_PERSON = rf"{_TOKEN}(?:\s+{_MID}){{0,2}}\s+{_TOKEN}"
# One or more people joined by "and" / "," / "&".
_PEOPLE = rf"{_PERSON}(?:\s*(?:,|and|&)\s*{_PERSON})*"

_WORK = (
    r"(?:[Ss]ermons?|[Ll]essons?|[Cc]hapters?|[Bb]ooks?|[Ee]ssays?"
    r"|[Pp]resentations?|[Tt]alks?)"
)

# Matched against the opening of the sermon text.
_HEAD_PATTERNS = [re.compile(p) for p in [
    rf"^A\s+({_PERSON})\s+{_WORK}\s+Summary",
    rf"^({_PERSON})\s+{_WORK}\s+Summary",
    # "A Benedict Kwok Hung-Biu Sermon Summary" — a longer name is safe here
    # because "Sermon Summary" closes the match, so it cannot run into the body.
    rf"^A\s+({_TOKEN}(?:\s+{_TOKEN}){{1,3}})\s+{_WORK}\s+Summary",
    rf"^A\s+{_WORK}\s+Summary\s+(?:from|by)\s+({_PEOPLE})",
    rf"^A\s+Summary\s+of\s+(?:an?\s+)?(?:\w+\s+)?by\s+({_PEOPLE})",
    rf"^Summaries?\s+of\s+{_WORK}\s+by\s+({_PEOPLE})",
    rf"^{_WORK}\s+from\s+({_PEOPLE})",
    rf"^A\s+(?:Lesson|Summary)\s+from\s+({_PEOPLE})",
    rf"^of\s+(?:an?\s+)?{_WORK}\s+(?:from|by)\s+({_PEOPLE})",
    rf"^of\s+(?:an?\s+)?({_PERSON})\s+{_WORK}\b",
    rf"^from\s+[\w\s]{{0,30}}?of\s+(?:an?\s+)?{_WORK}\s+by\s+({_PEOPLE})",
    rf"^from\s+[\w\s]{{0,30}}?of\s+(?:an?\s+)?({_PERSON})\s+{_WORK}\b",
    # Last resort: a "by <Name>" credit anywhere in the opening lines, which
    # covers book citations such as "in Mere Christianity by C.S. Lewis".
    rf"\bby\s+({_PEOPLE})",
]]

# Matched against the closing of the sermon text, where attribution often sits
# next to the source link.
_TAIL_PATTERNS = [re.compile(p) for p in [
    rf"from\s+an?\s+({_PERSON})\s+[Ss]ermon",
    rf"[Ss]ermons?\s+by\s+({_PEOPLE})",
    rf"\sby\s+({_PERSON})\s*(?:https?://)",
]]

# Capitalised words that start a sentence but are never an attribution.
_NOT_A_NAME = frozenset([
    "a", "an", "the", "sermon", "sermons", "summary", "summaries", "lesson",
    "lessons", "chapter", "book", "essay", "god", "gods", "christ", "jesus",
    "bible", "christian", "christians", "holy", "spirit", "lord", "father",
    "son", "many", "this", "that", "these", "those", "when", "what", "why",
    "how", "since", "first", "second", "third", "having", "he", "she", "they",
    "we", "it", "in", "on", "of", "to", "and", "but", "titled",
])

# Spellings of the same person that no rule can reconcile.
_CANONICAL = {
    "charles r. swindoll": "Charles Swindoll",
    "cs lewis": "C.S. Lewis",
    "c.s. lewis": "C.S. Lewis",
    "c. s. lewis": "C.S. Lewis",
    "n t wright": "N.T. Wright",
    "n. t. wright": "N.T. Wright",
    "n.t. wright": "N.T. Wright",
    "wing y so": "Wing So",
}


def _clean_name(raw: str) -> str:
    name = raw.strip().strip(".,;:'’").strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"['’]s$", "", name)

    if not name or len(name) > 60:
        return ""

    words = name.split()
    if not (2 <= len(words) <= 4):
        return ""

    if any(w.lower().replace(".", "") in _NOT_A_NAME for w in words):
        return ""

    # A full word ending in a period is a sentence break, not a name: the
    # "by <Name>" fallback would otherwise read "verified by Hubble. Then the
    # Big Bang..." as an author. Initials ("C.S.", "N.T.", "A.W.") are fine.
    if any(w.endswith(".") and len(w.replace(".", "")) > 2 for w in words):
        return ""

    return _CANONICAL.get(name.lower(), name)


def _split_people(phrase: str) -> list[str]:
    parts = re.split(r"\s*(?:,|\band\b|&)\s*", phrase)
    names = []
    for part in parts:
        name = _clean_name(part)
        if name and name not in names:
            names.append(name)
    return names


def extract_authors(text: str) -> list[str]:
    """All people credited in a sermon summary, most prominent first."""
    text = (text or "").strip()
    if not text:
        return []

    head = text[:300]
    for pattern in _HEAD_PATTERNS:
        m = pattern.search(head)
        if m:
            names = _split_people(m.group(1))
            if names:
                return names

    tail = text[-300:]
    for pattern in _TAIL_PATTERNS:
        m = pattern.search(tail)
        if m:
            names = _split_people(m.group(1))
            if names:
                return names

    return []


# -------------------- Catalog --------------------

_STOP_WORDS = frozenset([
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "is", "are", "be", "do", "does", "did", "have", "has",
    "i", "you", "he", "she", "it", "we", "they", "what", "how", "why", "when",
    "where", "about", "not", "this", "that", "your", "our", "all", "can",
    "will", "was", "were", "been", "had", "its", "him", "her", "them", "who",
    "their", "there",
])

# Words that describe the *act* of searching rather than the subject of the
# search. "sermons relating to raising children" is a query about raising
# children; leaving the first three words in drags every score down.
_QUERY_NOISE = frozenset([
    "sermon", "sermons", "message", "messages", "talk", "talks", "preach",
    "preaching", "sermonette", "relating", "related", "relate", "regarding",
    "concerning", "find", "finding", "show", "give", "get", "look", "looking",
    "search", "searching", "want", "wanted", "need", "please", "any", "anything",
    "some", "something", "list", "topic", "topics", "subject", "me", "my",
    "tell", "know", "like", "would", "could", "should", "more", "other",
    "others", "another", "else", "anymore", "again", "regard",
])


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", (text or "").lower())


def content_words(text: str) -> set:
    """Query words that carry the actual subject of the search."""
    words = {
        w for w in tokenize(text)
        if len(w) > 2 and w not in _STOP_WORDS and w not in _QUERY_NOISE
    }
    return words


def _matches_any(word: str, candidates: set) -> bool:
    """Word equality, loosened to a shared stem so "stewards" reaches Stewardship."""
    if word in candidates:
        return True
    if len(word) < 5:
        return False
    return any(
        len(c) >= 5 and (c.startswith(word[:5]) or word.startswith(c[:5]))
        for c in candidates
    )


def _name_spellings(name: str) -> set[str]:
    """Punctuation-free ways of writing a name: "C.S. Lewis" -> c s lewis, cs lewis."""
    parts = tokenize(name)
    if not parts:
        return set()

    spellings = {" ".join(parts)}

    joined, run = [], ""
    for part in parts:
        if len(part) == 1:
            run += part
        else:
            if run:
                joined.append(run)
                run = ""
            joined.append(part)
    if run:
        joined.append(run)
    spellings.add(" ".join(joined))

    return spellings


class Catalog:
    def __init__(self, sermons: list[dict]):
        self.sermons = sermons

        self.by_url = {s["url"]: s for s in sermons if s.get("url")}

        self.by_category: dict[str, list[dict]] = {}
        self.by_author: dict[str, list[dict]] = {}
        self.author_display: dict[str, str] = {}

        for sermon in sermons:
            category = (sermon.get("category") or "").strip()
            if category:
                self.by_category.setdefault(category.lower(), []).append(sermon)

            for author in sermon.get("authors", []):
                key = author.lower()
                self.by_author.setdefault(key, []).append(sermon)
                self.author_display.setdefault(key, author)

        # Spellings a person might type for each author. Punctuation is dropped
        # so "C.S. Lewis" is reachable as "c s lewis", and runs of initials are
        # also joined so the far more common "cs lewis" hits the same author.
        self.author_spellings: dict[str, str] = {}
        for key in self.by_author:
            for spelling in _name_spellings(key):
                self.author_spellings.setdefault(spelling, key)

        # Surname (and forename) lookups so "Edwards" and "Keller" work as well
        # as "Jonathan Edwards". A part shared by two authors maps to both,
        # which is what someone typing "Wright" wants.
        self.author_parts: dict[str, set[str]] = {}
        for key in self.by_author:
            for part in tokenize(key):
                if len(part) >= 4:
                    self.author_parts.setdefault(part, set()).add(key)

        # Category names reduced to the words a person would actually type.
        self.category_words: dict[str, set[str]] = {
            key: {w for w in tokenize(key) if w not in _STOP_WORDS}
            for key in self.by_category
        }

    # ---------- lookups ----------

    def find_authors(self, query: str) -> list[str]:
        """Author keys named by the query, or [] if it is not an author search."""
        text = " ".join(tokenize(query))
        if not text:
            return []

        # Full name spelled out: unambiguous, accept at any query length.
        full = []
        for spelling, key in self.author_spellings.items():
            if spelling in text and key not in full:
                full.append(key)
        if full:
            # Prefer the longest match so "tim keller" does not also report a
            # hypothetical "keller".
            full.sort(key=len, reverse=True)
            longest = full[0]
            return [k for k in full if k == longest or k not in longest]

        # A bare surname is only treated as an author search when the query is
        # short or explicitly asks for an author, so "peace" style topic words
        # never get captured.
        tokens = text.split()
        asks_for_author = bool(
            {"by", "author", "pastor", "preacher", "speaker"} & set(tokens)
        )
        if len(tokens) > 4 and not asks_for_author:
            return []

        matched: list[str] = []
        for token in tokens:
            for key in self.author_parts.get(token, ()):
                if key not in matched:
                    matched.append(key)
        return matched

    def find_category(self, query: str) -> str | None:
        """
        The category a query names, or None.

        Two conditions, and both matter. Every word of the category name has to
        appear in the query, so "what is love" is left to ordinary search rather
        than being pulled into "Gods Love". And the query must ask for nothing
        beyond the category, so "how do I share my faith with friends" stays a
        question about sharing faith instead of becoming a browse of the Faith
        category.
        """
        words = set(tokenize(query))
        if not words:
            return None

        asked_for = content_words(query)

        best = None
        for key, cat_words in self.category_words.items():
            if not cat_words:
                continue
            # Every word of the category name is present...
            if not all(_matches_any(w, words) for w in cat_words):
                continue
            # ...and the query asks for nothing else.
            if any(not _matches_any(w, cat_words) for w in asked_for):
                continue
            if best is None or len(cat_words) > len(self.category_words[best]):
                best = key
        return best

    def category_display(self, key: str) -> str:
        sermons = self.by_category.get(key, [])
        return sermons[0]["category"] if sermons else key.title()

    def keyword_search(self, query: str, limit: int = 10, min_score: float = 0.0) -> list[dict]:
        """
        Local title-and-text scan, used when the vector search comes back empty.

        A title hit is worth far more than a body hit: someone asking about
        children means "DISCIPLINE Your Children", not a sermon that happens to
        mention "children of God" in passing.
        """
        words = content_words(query)
        if not words:
            return []

        scored = []
        for sermon in self.sermons:
            title_words = set(tokenize(sermon["title"]))
            body_words = sermon["_body_words"]

            title_hits = len(words & title_words)
            body_hits = len(words & body_words)
            if not title_hits and not body_hits:
                continue

            score = (title_hits / len(words)) * 2.0 + (body_hits / len(words))
            if score < min_score:
                continue
            scored.append((score, sermon))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [sermon for _, sermon in scored[:limit]]


def _resolve_possessives(sermons: list[dict]) -> None:
    """
    Fold "Tim Kellers" into "Tim Keller", in place.

    Apostrophes were stripped from the corpus during ingestion, so possessive
    attributions ("summarized from a Tim Kellers sermon") extract with a
    trailing s. Dropping that s unconditionally would break real names, so a
    name is only folded when the shortened form is itself credited elsewhere in
    the corpus — which "Francis Collin" never is.
    """
    known = {a for s in sermons for a in s["authors"]}
    aliases = {a: a[:-1] for a in known if a.endswith("s") and a[:-1] in known}
    if not aliases:
        return

    logger.info(f"Folded possessive author names: {sorted(aliases)}")
    for sermon in sermons:
        folded = []
        for author in sermon["authors"]:
            author = aliases.get(author, author)
            if author not in folded:
                folded.append(author)
        sermon["authors"] = folded


def _load_sermons() -> list[dict]:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    sermons = []
    for title, entry in raw.items():
        url = (entry.get("url") or "").strip()
        if not url:
            continue
        content = entry.get("content") or ""
        sermons.append({
            "title": title.replace('"', "").strip(),
            "url": url,
            "category": (entry.get("category") or "").strip(),
            "authors": extract_authors(content),
            "_body_words": set(tokenize(content)),
        })

    _resolve_possessives(sermons)
    return sermons


def load_catalog() -> Catalog:
    try:
        catalog = Catalog(_load_sermons())
    except Exception as e:
        logger.error(f"Could not load sermon catalog: {e}")
        return Catalog([])

    logger.info(
        f"Catalog loaded: {len(catalog.sermons)} sermons, "
        f"{len(catalog.by_author)} authors, {len(catalog.by_category)} categories"
    )
    return catalog
