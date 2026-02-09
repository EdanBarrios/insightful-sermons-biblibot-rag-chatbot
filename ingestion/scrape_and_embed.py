"""
Fixed sermon scraper for insightfulsermons.com
Scrapes INDIVIDUAL SERMONS from each category page using Selenium.
Runs daily via GitHub Actions.

Usage:
    python ingestion/scrape_and_embed.py
"""

import json
import os
import sys
import logging
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
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# Setup logging
log_dir = Path(__file__).parent
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Constants
BASE_URL = "https://www.insightfulsermons.com"
CATEGORIES_URL = f"{BASE_URL}/"
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Initialize services
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("sermon-index")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("✅ Services initialized")
except Exception as e:
    logger.error(f"❌ Initialization failed: {e}")
    sys.exit(1)

# Cleaning utilities
def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def clean_content(content):
    """Clean sermon content"""
    bible_books = "Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth|1 Samuel|2 Samuel|1 Kings|2 Kings|1 Chronicles|2 Chronicles|Ezra|Nehemiah|Esther|Job|Psalms|Proverbs|Ecclesiastes|Song of Solomon|Isaiah|Jeremiah|Lamentations|Ezekiel|Daniel|Hosea|Joel|Amos|Obadiah|Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi|Matthew|Mark|Luke|John|Acts|Romans|1 Corinthians|2 Corinthians|Galatians|Ephesians|Philippians|Colossians|1 Thessalonians|2 Thessalonians|1 Timothy|2 Timothy|Titus|Philemon|Hebrews|James|1 Peter|2 Peter|1 John|2 John|3 John|Jude|Revelation"
    
    content = re.sub(r'\b(' + bible_books + r')\s+\d+[:]?\d*(-\d+)?', '', content)
    content = re.sub(r'\d+', '', content)
    content = re.sub(r'\[.*?\]', '', content)
    content = re.sub(r'\(.*?\)', '', content)
    content = re.sub(r'[:;,\-]', '', content)
    content = re.sub(r'^(.*?\bSummary\b)', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\b(Summary|Summarized).*$', '', content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r'\s+', ' ', content).strip()
    return content

def generate_doc_id(title, url):
    """Generate stable document ID"""
    combined = f"{title}|{url}"
    return hashlib.md5(combined.encode()).hexdigest()

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def scrape_sermons():
    """Scrape individual sermons from category pages"""
    logger.info("🔄 Starting sermon scraping...")
    
    # Setup Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
    except Exception as e:
        logger.warning(f"Headless Chrome failed, trying normal mode: {e}")
        try:
            driver = webdriver.Chrome()
        except Exception as e2:
            logger.error(f"❌ Could not initialize Chrome: {e2}")
            sys.exit(1)
    
    articles = {}
    
    try:
        # Go to home page to get categories
        logger.info(f"Loading home page: {CATEGORIES_URL}")
        driver.get(CATEGORIES_URL)
        time.sleep(3)
        
        # Find all category links in the menu
        try:
            # Look for menu items with sermon category links
            category_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='.html'][class*='menu']")
            if not category_links:
                category_links = driver.find_elements(By.CSS_SELECTOR, "a[href$='.html']")
        except:
            logger.warning("Could not find category links, trying alternative selector")
            category_links = driver.find_elements(By.CSS_SELECTOR, "a[href]")
        
        # Extract category URLs and names before looping (avoid stale elements)
        category_data = []
        for link in category_links:
            try:
                url = link.get_attribute('href')
                name = link.text.strip()
                
                # Filter for actual category pages (not images, not nav items, etc)
                if url and name and len(name) > 2 and url.endswith('.html') and not url.startswith('javascript'):
                    if url.startswith('/'):
                        url = BASE_URL + url
                    if url not in [c[0] for c in category_data]:  # Avoid duplicates
                        category_data.append((url, name))
                        logger.info(f"  Found category: {name} -> {url}")
            except:
                pass
        
        logger.info(f"Found {len(category_data)} categories")
        
        # Now scrape each category's sermons
        for category_url, category_name in category_data:
            try:
                logger.info(f"\n📂 Scraping category: {category_name}")
                logger.info(f"   URL: {category_url}")
                
                driver.get(category_url)
                time.sleep(2)
                
                # Find all sermon links on this category page
                # Look for menu subitems which are individual sermons
                sermon_links = []
                
                # Try multiple selectors to find sermon links
                for selector in [
                    "a[href$='.html'][class*='menu-subitem']",  # Menu items
                    "a[href*='/'][class*='sermon']",  # Sermon pages
                    "li a[href$='.html']",  # List items
                    "a[href$='.html']"  # Any .html link
                ]:
                    try:
                        found = driver.find_elements(By.CSS_SELECTOR, selector)
                        if found:
                            logger.info(f"   Found {len(found)} links with selector: {selector}")
                            for elem in found:
                                href = elem.get_attribute('href')
                                text = elem.text.strip()
                                if href and text and len(text) > 3 and href.endswith('.html'):
                                    if href.startswith('/'):
                                        href = BASE_URL + href
                                    sermon_links.append((href, text))
                            break
                    except:
                        pass
                
                # Remove duplicates while preserving order
                seen = set()
                unique_sermons = []
                for href, text in sermon_links:
                    if href not in seen:
                        seen.add(href)
                        unique_sermons.append((href, text))
                
                logger.info(f"   Got {len(unique_sermons)} unique sermons")
                
                # Now scrape content from each sermon
                for sermon_url, sermon_title in unique_sermons:
                    try:
                        logger.info(f"     Scraping: {sermon_title[:40]}...")
                        
                        driver.get(sermon_url)
                        time.sleep(1)
                        
                        # Get page content
                        try:
                            # Look for paragraph/content elements
                            paragraphs = driver.find_elements(By.CSS_SELECTOR, 'p, .paragraph, .content, article p')
                            content = " ".join([remove_non_ascii(p.text) for p in paragraphs if p.text.strip()])
                        except:
                            content = ""
                        
                        if not content:
                            # Fallback: get all text from body
                            try:
                                body = driver.find_element(By.TAG_NAME, 'body')
                                content = remove_non_ascii(body.text)
                            except:
                                content = ""
                        
                        content = clean_content(content)
                        
                        # Store if we got meaningful content
                        if content and len(content) > 100:
                            articles[sermon_title] = {
                                "content": content,
                                "url": sermon_url,
                                "category": category_name
                            }
                            logger.info(f"       ✅ Stored: {sermon_title[:40]}...")
                        else:
                            logger.warning(f"       ⚠️  Content too short for: {sermon_title[:40]}...")
                    
                    except Exception as e:
                        logger.warning(f"       ❌ Error scraping sermon: {e}")
                        continue
            
            except Exception as e:
                logger.error(f"   ❌ Error scraping category {category_name}: {e}")
                continue
        
        logger.info(f"\n✅ Scraping complete! Got {len(articles)} sermons")
        
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.quit()
    
    return articles

def embed_and_upsert(articles):
    """Embed sermon chunks and upsert to Pinecone"""
    logger.info("📊 Embedding and upserting to Pinecone...")
    
    vectors = []
    
    for title, article in articles.items():
        try:
            content = article['content']
            url = article['url']
            category = article['category']
            
            chunks = chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                doc_id = f"{generate_doc_id(title, url)}_chunk_{i}"
                embedding = embedder.encode(chunk).tolist()
                
                vectors.append({
                    "id": doc_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "title": title,
                        "url": url,
                        "category": category,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                })
        except Exception as e:
            logger.warning(f"Error processing {title}: {e}")
            continue
    
    logger.info(f"Created {len(vectors)} vectors from {len(articles)} sermons")
    
    if not vectors:
        logger.warning("⚠️ No vectors to upload!")
        return
    
    # Upsert in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
            logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
        except Exception as e:
            logger.error(f"Error upserting batch: {e}")
    
    logger.info("✅ All vectors upserted")

def main():
    """Main ingestion pipeline"""
    logger.info("=" * 60)
    logger.info(f"Starting ingestion at {datetime.now()}")
    logger.info("=" * 60)
    
    try:
        articles = scrape_sermons()
        
        if not articles:
            logger.warning("⚠️ No articles scraped")
            return
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_file = DATA_DIR / f"sermon_data_{timestamp}.json"
        
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Saved raw data to {data_file}")
        
        # Embed and upsert
        embed_and_upsert(articles)
        
        # Final stats
        try:
            stats = index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            logger.info(f"📊 Total vectors in Pinecone: {total_vectors}")
        except:
            pass
        
        logger.info("=" * 60)
        logger.info("✅ Ingestion complete!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()