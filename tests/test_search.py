"""
Search routing verification.

Checks the parts of lookup that do not need Pinecone: author extraction, author
and category routing, and follow-up bookkeeping. Every case here is one the bot
got wrong in production at some point.

Usage:
    python tests/test_search.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.catalog import extract_authors, load_catalog
from app.search import previously_shown


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'


failures = []


def check(name, actual, expected):
    passed = actual == expected
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"{status} | {name}")
    if not passed:
        print(f"     expected: {expected!r}")
        print(f"     actual:   {actual!r}")
        failures.append(name)


def test_author_extraction():
    print(f"\n{Colors.BLUE}=== Author extraction ==={Colors.END}")

    cases = [
        ("A Tim Keller Sermon Summary Good behavior is the crux of many.", ["Tim Keller"]),
        ("Sermons from Zac Poonen God breathed into us a conscience.", ["Zac Poonen"]),
        ("A Philip Yancey Chapter Summary Jesus was a friend to sinners.", ["Philip Yancey"]),
        ("of a David Pao Sermon Christians rest their hope beyond this world.", ["David Pao"]),
        ("A Benedict Kwok Hung-Biu Sermon Summary Curiously, the Bible uses.",
         ["Benedict Kwok Hung-Biu"]),
        ("Sermons from John Ortberg and Tim Keller We are made for each other.",
         ["John Ortberg", "Tim Keller"]),
        ("of Book Two in Mere Christianity by C.S. Lewis Many believe that.", ["C.S. Lewis"]),
        # The attribution runs straight into the body; the name must stop at
        # the surname rather than swallowing the next sentence.
        ("A Lesson from Tim Keller If Christ is who He said He is, then.", ["Tim Keller"]),
        # A sentence break is not an author.
        ("Einstein predicted the universe was expanding, verified by Hubble. "
         "Then the Big Bang theory postulated more.", []),
        # No attribution at all.
        ("A Sermon Summary After Christ had fed 5000, many wanted a king.", []),
    ]
    for text, expected in cases:
        check(f"extract {text[:44]!r}", extract_authors(text), expected)


def test_author_routing(catalog):
    print(f"\n{Colors.BLUE}=== Author routing ==={Colors.END}")

    def names(query):
        return [catalog.author_display[k] for k in catalog.find_authors(query)]

    check("'Jonathan Edwards'", names("Jonathan Edwards"), ["Jonathan Edwards"])
    check("'Edwards'", names("Edwards"), ["Jonathan Edwards"])
    check("'sermons by Keller'", names("sermons by Keller"), ["Tim Keller"])
    check("'CS Lewis' matches 'C.S. Lewis'", names("CS Lewis"), ["C.S. Lewis"])
    check("topic query is not an author", names("what is stewardship"), [])
    check("'love' is not an author", names("what is love"), [])

    # Peter reported one Jonathan Edwards sermon and needed coaxing for more.
    edwards = catalog.by_author["jonathan edwards"]
    check("Jonathan Edwards has more than one sermon", len(edwards) > 1, True)


def test_possessive_names(catalog):
    print(f"\n{Colors.BLUE}=== Possessive names ==={Colors.END}")

    for bad in ["tim kellers", "john ortbergs", "rick warrens"]:
        check(f"{bad!r} folded away", bad in catalog.by_author, False)

    # ...without breaking a real name that ends in s.
    check("'Francis Collins' kept", "francis collins" in catalog.by_author, True)


def test_category_routing(catalog):
    print(f"\n{Colors.BLUE}=== Category routing ==={Colors.END}")

    check("'What is stewardship'", catalog.find_category("What is stewardship"), "stewardship")
    check("'stewardship'", catalog.find_category("stewardship"), "stewardship")
    check("'sermons about hope'", catalog.find_category("sermons about hope"), "hope")
    check("'prayer'", catalog.find_category("prayer"), "prayer")

    # "Gods Love" must not swallow a general question about love.
    check("'What is love' is not a category", catalog.find_category("What is love"), None)
    # Nor may a real question be turned into a category browse.
    check(
        "'how do I share my faith with friends' is not a category",
        catalog.find_category("how do I share my faith with friends"),
        None,
    )
    check("Stewardship is populated", len(catalog.by_category["stewardship"]) > 0, True)


def test_keyword_fallback(catalog):
    print(f"\n{Colors.BLUE}=== Keyword fallback ==={Colors.END}")

    titles = [s["title"] for s in catalog.keyword_search("sermons relating to raising children")]
    check(
        "'raising children' reaches the children sermons",
        any("Children" in t for t in titles),
        True,
    )
    check(
        "nonsense query stays unanswered",
        catalog.keyword_search("best pizza recipe", min_score=0.9),
        [],
    )


def test_previously_shown():
    print(f"\n{Colors.BLUE}=== Follow-up bookkeeping ==={Colors.END}")

    history = [
        {"role": "user", "content": "Jonathan Edwards"},
        {"role": "assistant", "content":
            "Here are 2 sermons:\n\nThe Wrath of God\n"
            "[Read the sermon](https://www.insightfulsermons.com/the-wrath-of-god.html)\n\n"
            "Esau I Hated\n"
            "[Read the sermon](https://www.insightfulsermons.com/esau-i-hated.html)"},
        {"role": "user", "content": "more"},
    ]
    check(
        "links already sent are collected",
        previously_shown(history),
        {
            "https://www.insightfulsermons.com/the-wrath-of-god.html",
            "https://www.insightfulsermons.com/esau-i-hated.html",
        },
    )
    check("user messages are ignored", previously_shown(history[:1]), set())


def main():
    print(f"\n{Colors.BLUE}{'=' * 50}{Colors.END}")
    print(f"{Colors.BLUE}BibliBot Search Verification{Colors.END}")
    print(f"{Colors.BLUE}{'=' * 50}{Colors.END}")

    catalog = load_catalog()
    if not catalog.sermons:
        print(f"{Colors.RED}Could not load the sermon catalog.{Colors.END}")
        return 1

    print(
        f"\nCatalog: {len(catalog.sermons)} sermons, "
        f"{len(catalog.by_author)} authors, {len(catalog.by_category)} categories"
    )

    test_author_extraction()
    test_author_routing(catalog)
    test_possessive_names(catalog)
    test_category_routing(catalog)
    test_keyword_fallback(catalog)
    test_previously_shown()

    print(f"\n{Colors.BLUE}=== Summary ==={Colors.END}")
    if failures:
        print(f"{Colors.RED}✗ {len(failures)} check(s) failed:{Colors.END}")
        for name in failures:
            print(f"    {name}")
        return 1

    print(f"{Colors.GREEN}🎉 All checks passed.{Colors.END}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
