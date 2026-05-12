"""
Daily sermon scraper for insightfulsermons.com
Automatically detects and uploads NEW sermons that haven't been indexed yet.

Runs daily via GitHub Actions to keep Pinecone in sync with the website.

Usage:
    python ingestion/scrape_and_embed.py
"""

import json
import os
import random
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
import hashlib
import re
import time

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


# ----------------------------
# Logging
# ----------------------------
log_dir = Path(__file__).parent
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(log_dir / "ingestion.log", maxBytes=1_000_000, backupCount=3),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ----------------------------
# Env / constants
# ----------------------------
load_dotenv()

BASE_URL = "https://www.insightfulsermons.com"
CATEGORIES_URL = f"{BASE_URL}/categories.html"
MAX_RETRIES = 3
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ----------------------------
# Services
# ----------------------------
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("sermon-index")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("✅ Services initialized")
except Exception as e:
    logger.error(f"❌ Initialization failed: {e}")
    sys.exit(1)


def remove_non_ascii(text: str) -> str:
    """Remove non-ASCII characters"""
    return re.sub(r"[^\x00-\x7F]+", "", text or "")


def clean_content(content: str) -> str:
    """
    Clean sermon content for embeddings/search.
    Keep meaning; avoid aggressive deletion that can wipe sermons.
    """
    if not content:
        return ""

    # Remove bracketed footnotes / junk
    content = re.sub(r"\[.*?\]", " ", content)

    # Remove ONLY a leading Summary/Summarized label (not the whole rest)
    content = re.sub(r"^\s*(summary|summarized)\s*:?\s*", "", content, flags=re.IGNORECASE)

    # Normalize whitespace
    content = re.sub(r"\s+", " ", content).strip()
    return content


_AUTHOR_PATTERNS = [
    re.compile(r"^A\s+(.+?)\s+Sermon\s+Summary", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Sermons?\s+from\s+(.+?)(?:\s*\n|\s{2,}|$)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^A\s+Lesson\s+from\s+(.+?)(?:\s*\n|\s{2,}|$)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"Summarized\s+from\s+(?:a\s+)?(?:sermon\s+(?:by|from)\s+)?(.+?)(?:\s*[,.]|\s*https?://|$)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"Summary\s+from\s+(?:a?\s+)?(?:sermon\s+(?:by|from)\s+)?(.+?)(?:\s*[,.]|\s*https?://|$)", re.IGNORECASE | re.MULTILINE),
]


def extract_author(content: str) -> str | None:
    """Extract author name from known content attribution patterns."""
    for pattern in _AUTHOR_PATTERNS:
        m = pattern.search(content.strip())
        if m:
            author = m.group(1).strip().rstrip(".,")
            if 3 <= len(author) <= 60:
                return author
    return None


def generate_doc_id(url: str):
    return hashlib.md5(url.encode()).hexdigest()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks (word-based)"""
    words = (text or "").split()
    chunks: list[str] = []
    if not words:
        return chunks

    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def load_existing_sermons(json_file: str) -> dict:
    """Load existing sermon URLs from local JSON"""
    try:
        if not os.path.exists(json_file):
            return {}
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load existing sermons: {e}")
        return {}


def save_sermons(sermon_data: dict, json_file: str) -> None:
    """Save sermon data to JSON file"""
    try:
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sermon_data, f, ensure_ascii=False, indent=4)
        logger.info(f"💾 Saved {len(sermon_data)} sermons to {json_file}")
    except Exception as e:
        logger.error(f"Error saving sermons: {e}")


def get_link_text(element, url: str) -> str:
    """Extract text from link - fallback to URL if text is empty"""
    try:
        title_span = element.find_element(By.CSS_SELECTOR, ".wsite-menu-title")
        text = title_span.text.strip()
        if text:
            return text
    except Exception:
        pass

    if url:
        name = url.replace("https://www.insightfulsermons.com/", "").replace(".html", "")
        name = name.replace("-", " ").title()
        return name

    return ""


def scrape_sermons(existing_sermons: dict | None = None) -> dict[str, dict]:
    """
    Scrape sermons from the website.

    IMPORTANT: Every page on this site renders the full left-nav menu (hundreds of links).
    Fix approach:
      - Only treat /categories.html as the source of CATEGORY links.
      - For each category page, only collect SERMON links that are children of that category
        in the nav hierarchy (not the entire site menu).
      - Deduplicate by URL and never follow links discovered on sermon pages.
      - Extract sermon body from #wsite-content div.paragraph (Weebly layout).
    """
    logger.info("🔄 Starting sermon scraping...")

    existing_sermons = existing_sermons or {}
    existing_urls = {
        v.get("url")
        for v in existing_sermons.values()
        if isinstance(v, dict) and v.get("url")
    }

    visited_sermon_urls: set[str] = set()

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    try:
        driver = webdriver.Chrome(options=chrome_options)
    except Exception as e:
        logger.error(f"❌ Could not initialize Chrome: {e}")
        sys.exit(1)

    def _abs_url(href: str) -> str:
        if not href:
            return ""
        href = href.strip()

        # Normalize /home/ prefix
        href = href.replace("/home/", "/")
        href = href.replace("https://www.insightfulsermons.com/home/", "https://www.insightfulsermons.com/")
        href = href.replace("http://www.insightfulsermons.com/home/", "http://www.insightfulsermons.com/")

        if href.startswith("/"):
            return BASE_URL + href
        if href.startswith("http://") or href.startswith("https://"):
            return href
        if href.endswith(".html"):
            return f"{BASE_URL}/{href.lstrip('/')}"
        return href


    def _get_sermon_title() -> str:
        """Try several in-page selectors for a sermon title."""
        for css in ["h1", "h2", ".wsite-content-title", ".wsite-section-title"]:
            try:
                el = driver.find_element(By.CSS_SELECTOR, css)
                t = (el.text or "").strip()
                if t and len(t) > 2:
                    return t
            except Exception:
                continue
        return ""

    def _get_main_text() -> str:
        """
        Weebly pages store real text in div.paragraph blocks under #wsite-content.
        Pull those first (best signal, avoids menu junk).
        """
        try:
            content_root = driver.find_element(By.ID, "wsite-content")

            paras = content_root.find_elements(By.CSS_SELECTOR, "div.paragraph")
            parts = [p.text.strip() for p in paras if p.text and p.text.strip()]
            if parts:
                return "\n\n".join(parts).strip()

            # Secondary: sometimes content is in wsite-elements without div.paragraph
            els = content_root.find_elements(By.CSS_SELECTOR, ".wsite-elements, .wsite-section-elements")
            parts = [e.text.strip() for e in els if e.text and e.text.strip()]
            txt = "\n\n".join(parts).strip()
            if txt:
                return txt

            # Tertiary: plain <p> tags
            p_els = content_root.find_elements(By.CSS_SELECTOR, "p")
            parts = [p.text.strip() for p in p_els if p.text and p.text.strip()]
            txt = "\n\n".join(parts).strip()
            if txt and len(txt) > 200:
                return txt

            return (content_root.text or "").strip()
        except Exception:
            return ""

    all_sermons: dict[str, dict] = {}
    failed_urls: list[tuple[str, str]] = []

    try:
        # Step 1: Load /categories.html and collect CATEGORY links only.
        logger.info(f"📂 Loading categories page: {CATEGORIES_URL}")
        driver.get(CATEGORIES_URL)
        time.sleep(1.5)

        category_els = driver.find_elements(
            By.XPATH,
            "//a[contains(@class,'wsite-menu-subitem') and .//span[contains(@class,'wsite-menu-arrow')]]",
        )

        category_links: list[tuple[str, str]] = []
        for el in category_els:
            href = _abs_url(el.get_attribute("href"))
            title = get_link_text(el, href)
            if href and href.endswith(".html") and href != CATEGORIES_URL:
                category_links.append((href, title))

        # De-dupe categories by URL (some templates duplicate nav items)
        seen_cat: set[str] = set()
        deduped_cats: list[tuple[str, str]] = []
        for u, t in category_links:
            if u not in seen_cat:
                seen_cat.add(u)
                deduped_cats.append((u, t))
        category_links = deduped_cats

        logger.info(f"✅ Found {len(category_links)} category links")
        category_url_set = {u for (u, _) in category_links}

        # Step 2: For each category page, collect ONLY the sermon links that are children of that category.
        sermon_links: list[tuple[str, str, str]] = []  # (sermon_url, sermon_title, category_title)

        for cat_url, cat_title in category_links:
            try:
                driver.get(cat_url)
                time.sleep(1.0 + random.uniform(-0.1, 0.2))

                # Find the nav anchor that matches this category, then pull its descendant sermon items.
                cat_anchor = None
                try:
                    cat_anchor = driver.find_element(
                        By.XPATH,
                        f"//a[contains(@class,'wsite-menu-subitem') and @href='{cat_url}']",
                    )
                except Exception:
                    rel = cat_url.replace(BASE_URL, "")
                    try:
                        cat_anchor = driver.find_element(
                            By.XPATH,
                            f"//a[contains(@class,'wsite-menu-subitem') and @href='{rel}']",
                        )
                    except Exception:
                        cat_anchor = None

                sermon_anchors = []
                if cat_anchor is not None:
                    try:
                        li = cat_anchor.find_element(By.XPATH, "./ancestor::li[1]")
                        sermon_anchors = li.find_elements(
                            By.XPATH,
                            ".//a[contains(@class,'wsite-menu-subitem') and not(.//span[contains(@class,'wsite-menu-arrow')])]",
                        )
                    except Exception:
                        sermon_anchors = []

                # Fallback (filtered): only take in-content links that are not categories/util pages
                if not sermon_anchors:
                    candidates = driver.find_elements(By.CSS_SELECTOR, "#wsite-content a[href$='.html']")
                    for a in candidates:
                        href = _abs_url(a.get_attribute("href"))
                        if not href or not href.endswith(".html"):
                            continue
                        if href in category_url_set:
                            continue
                        if href == CATEGORIES_URL or href == cat_url:
                            continue
                        if href.endswith("/categories.html"):
                            continue

                        stitle = (get_link_text(a, href) or "").strip()
                        if not stitle or len(stitle) < 3:
                            continue

                        sermon_anchors.append(a)

                for a in sermon_anchors:
                    href = _abs_url(a.get_attribute("href"))
                    if not href or not href.endswith(".html"):
                        continue
                    if href == CATEGORIES_URL or href == cat_url:
                        continue
                    if href.endswith("/categories.html"):
                        continue
                    if href in category_url_set:
                        continue

                    stitle = (get_link_text(a, href) or "").strip()
                    if not stitle or len(stitle) < 2:
                        continue

                    sermon_links.append((href, stitle, cat_title or "General"))

            except Exception as e:
                logger.warning(f"Error reading category {cat_title} ({cat_url}): {e}")
                continue

        # De-dupe sermon URLs globally
        seen_ser: set[str] = set()
        deduped_ser: list[tuple[str, str, str]] = []
        for u, t, c in sermon_links:
            if u not in seen_ser:
                seen_ser.add(u)
                deduped_ser.append((u, t, c))
        sermon_links = deduped_ser

        logger.info(f"✅ Collected {len(sermon_links)} unique sermon links across categories")

        # Step 3: Visit each sermon once and scrape main content.
        skipped_existing = 0
        scraped = 0

        for sermon_url, sermon_title, cat_title in sermon_links:
            if sermon_url in visited_sermon_urls:
                continue
            visited_sermon_urls.add(sermon_url)

            if sermon_url in existing_urls:
                skipped_existing += 1
                continue

            logger.info(f"📖 Scraping sermon: {sermon_title[:80]} ({cat_title})")

            page_loaded = False
            for attempt in range(MAX_RETRIES):
                try:
                    driver.get(sermon_url)
                    time.sleep(1.0 + random.uniform(-0.1, 0.2))
                    page_loaded = True
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"  ↻ Retry {attempt + 1}/{MAX_RETRIES}: {e}")
                        time.sleep(1.5 * (attempt + 1))
                    else:
                        logger.warning(f"  ❌ Load failed after {MAX_RETRIES} attempts: {e}")
                        failed_urls.append((sermon_url, f"Load failed: {e}"))

            if not page_loaded:
                continue

            try:
                page_title = _get_sermon_title()
                final_title = page_title or sermon_title

                raw_content = _get_main_text()
                raw_content = remove_non_ascii(raw_content)
                content = clean_content(raw_content)

                if not content or len(content) < 200:
                    logger.warning(
                        f"  ⚠️ Content too short (raw={len(raw_content)} cleaned={len(content)}) - skipping - {sermon_url}"
                    )
                    logger.warning(f"  RAW SNIP: {raw_content[:160]!r}")
                    failed_urls.append((sermon_url, f"Content too short ({len(content)} chars)"))
                    continue

                if len(content) > 50_000:
                    logger.warning(f"  ⚠️ Content unusually long ({len(content)} chars) - {sermon_url}")

                author = extract_author(content)

                key = final_title
                if key in all_sermons and all_sermons[key].get("url") != sermon_url:
                    key = f"{final_title} ({sermon_url.rsplit('/', 1)[-1].replace('.html','')})"

                all_sermons[key] = {
                    "content": content,
                    "url": sermon_url,
                    "category": cat_title or "General",
                    "author": author,
                }
                scraped += 1
                if scraped % 25 == 0:
                    logger.info(f"📊 Progress: {scraped} scraped, {len(failed_urls)} failed so far")
                logger.info("  ✅ Stored")

            except Exception as e:
                logger.warning(f"Error scraping sermon {sermon_title} ({sermon_url}): {e}")
                failed_urls.append((sermon_url, f"Extraction error: {e}"))
                continue

        logger.info("=" * 60)
        logger.info("✅ Scraping complete!")
        logger.info(f"   Skipped existing (by URL): {skipped_existing}")
        logger.info(f"   New sermons scraped: {scraped}")
        logger.info(f"   Total sermons collected this run: {len(all_sermons)}")
        logger.info(f"   Failed/skipped: {len(failed_urls)}")
        logger.info("=" * 60)
        if failed_urls:
            logger.warning(f"\n⚠️ Failed/skipped URLs ({len(failed_urls)}):")
            for furl, reason in failed_urls:
                logger.warning(f"  - {furl}: {reason}")

    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    return all_sermons


def embed_and_upsert(sermon_data: dict[str, dict]) -> bool:
    """Embed sermon chunks and upsert to Pinecone"""
    logger.info(f"📊 Embedding and upserting {len(sermon_data)} sermons...")

    BATCH_SIZE = 100
    pending: list[dict] = []
    total_upserted = 0

    def _flush() -> bool:
        nonlocal total_upserted
        if not pending:
            return True
        try:
            index.upsert(vectors=pending)
            total_upserted += len(pending)
            logger.info(f"Upserted batch — {total_upserted} vectors so far")
            pending.clear()
            return True
        except Exception as e:
            logger.error(f"Error upserting batch: {e}")
            return False

    for title, sermon in sermon_data.items():
        try:
            content = sermon.get("content", "")
            url = sermon.get("url", "")
            category = sermon.get("category", "General")
            author = sermon.get("author") or ""

            if not content or len(content) < 50:
                logger.debug(f"Skipping {title} - content too short")
                continue

            chunks = chunk_text(content)
            if not chunks:
                continue

            doc_id_base = generate_doc_id(url)
            embeddings = embedder.encode(chunks, batch_size=64, show_progress_bar=False)

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                pending.append(
                    {
                        "id": f"{doc_id_base}_chunk_{i}",
                        "values": embedding.tolist(),
                        "metadata": {
                            "text": chunk,
                            "title": title,
                            "url": url,
                            "category": category,
                            "author": author,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                        },
                    }
                )
                if len(pending) >= BATCH_SIZE:
                    if not _flush():
                        return False

        except Exception as e:
            logger.warning(f"Error processing {title}: {e}")
            continue

    if not _flush():
        return False

    if total_upserted == 0:
        logger.warning("⚠️ No vectors were uploaded!")
        return False

    logger.info(f"Created and upserted {total_upserted} vectors total")

    try:
        stats = index.describe_index_stats()
        total_vectors = stats.get("total_vector_count", 0)
        logger.info(f"✅ Total vectors in Pinecone: {total_vectors}")
    except Exception:
        pass

    return True


def main() -> None:
    """Main ingestion pipeline"""
    logger.info("=" * 60)
    logger.info(f"Starting daily ingestion at {datetime.now()}")
    logger.info("=" * 60)

    json_file = DATA_DIR / "sermon_data.json"

    existing_sermons = load_existing_sermons(str(json_file))
    logger.info(f"Loaded {len(existing_sermons)} existing sermons\n")

    current_sermons = scrape_sermons(existing_sermons)

    if not current_sermons:
        logger.error("❌ No sermons scraped from website!")
        sys.exit(1)

    new_sermons = {}
    updated_sermons = {}

    for title, sermon_data in current_sermons.items():
        if title not in existing_sermons:
            new_sermons[title] = sermon_data
        else:
            if sermon_data["url"] != existing_sermons[title].get("url"):
                updated_sermons[title] = sermon_data

    logger.info("\n📊 Sync Summary:")
    logger.info(f"   Existing sermons in file: {len(existing_sermons)}")
    logger.info(f"   Current on website: {len(current_sermons)}")
    logger.info(f"   New sermons: {len(new_sermons)}")
    logger.info(f"   Updated sermons: {len(updated_sermons)}")

    save_sermons(current_sermons, str(json_file))

    to_upsert = {**new_sermons, **updated_sermons}
    if not to_upsert:
        logger.info("✅ No new or updated sermons — Pinecone already up to date.")
        success = True
    else:
        logger.info(f"\n📤 Uploading {len(to_upsert)} new/updated sermons to Pinecone...")
        success = embed_and_upsert(to_upsert)

    if success:
        logger.info("=" * 60)
        logger.info("✅ Daily ingestion complete!")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("❌ Daily ingestion failed!")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()