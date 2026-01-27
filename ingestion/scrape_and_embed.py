"""
Fixed sermon scraper for insightfulsermons.com
Saves correct category URLs that actually work.

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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion/ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Constants
BASE_URL = "https://www.insightfulsermons.com"
CATEGORIES_URL = f"{BASE_URL}/categories.html"
DATA_DIR = Path("data")
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
    """Scrape sermons from website"""
    logger.info("🔍 Starting scraping...")
    
    # Setup Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
    except Exception as e:
        logger.warning(f"Headless mode failed, trying normal mode: {e}")
        driver = webdriver.Chrome()
    
    articles = {}
    
    try:
        # Go to categories page
        logger.info(f"Loading categories page: {CATEGORIES_URL}")
        driver.get(CATEGORIES_URL)
        time.sleep(3)
        
        # Find all category links and extract their data BEFORE looping
        category_links = driver.find_elements(By.CSS_SELECTOR, "a[href$='.html']")
        
        # Extract URLs and names upfront to avoid stale element references
        category_data = []
        for link in category_links:
            try:
                url = link.get_attribute('href')
                name = link.text.strip()
                if url and name and len(name) > 2:  # Valid category
                    category_data.append((url, name))
            except:
                pass
        
        logger.info(f"Found {len(category_data)} categories to scrape")
        
        # Now scrape each category (no stale references!)
        for category_url, category_name in category_data[:20]:  # Limit to 20
            try:
                logger.info(f"Scraping: {category_name} ({category_url})")
                
                driver.get(category_url)
                time.sleep(2)
                
                # Find posts
                posts = driver.find_elements(By.CSS_SELECTOR, "[class*='post']")
                
                if not posts:
                    logger.warning(f"No posts found for {category_name}")
                    continue
                
                logger.info(f"Found {len(posts)} posts")
                
                # Extract content from each post
                for post in posts:
                    try:
                        # Get title
                        title = None
                        for t_sel in ['h1', 'h2', 'h3', '[class*="title"]', 'a']:
                            try:
                                title_elem = post.find_element(By.CSS_SELECTOR, t_sel)
                                title = remove_non_ascii(title_elem.text.strip())
                                if title:
                                    break
                            except:
                                pass
                        
                        if not title:
                            continue
                        
                        # FIXED: Use the category URL (we know these work!)
                        # This ensures links will be valid
                        href = category_url
                        
                        # Get content
                        try:
                            paragraphs = post.find_elements(By.CSS_SELECTOR, 'p, .paragraph')
                            content = " ".join([remove_non_ascii(p.text) for p in paragraphs])
                        except:
                            content = remove_non_ascii(post.text)
                        
                        content = clean_content(content)
                        
                        if content and len(content) > 100:
                            articles[title] = {
                                "content": content,
                                "url": href,  # Category URL (guaranteed to work)
                                "category": category_name
                            }
                            logger.info(f"  ✅ Scraped: {title[:50]}...")
                    
                    except Exception as e:
                        logger.warning(f"Error scraping post: {e}")
                        continue
            
            except Exception as e:
                logger.error(f"Error scraping category {category_name}: {e}")
                continue
        
        logger.info(f"✅ Scraped {len(articles)} sermons total")
        
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.quit()
    
    return articles

def embed_and_upsert(articles):
    """Embed sermon chunks and upsert to Pinecone"""
    logger.info("🔄 Embedding and upserting...")
    
    vectors = []
    
    for title, article in articles.items():
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
                    "url": url,  # Now contains working category URL
                    "category": category,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
    
    logger.info(f"Created {len(vectors)} vectors from {len(articles)} sermons")
    
    if not vectors:
        logger.warning("No vectors to upload!")
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
    logger.info("=" * 50)
    logger.info(f"Starting ingestion at {datetime.now()}")
    logger.info("=" * 50)
    
    try:
        articles = scrape_sermons()
        
        if not articles:
            logger.warning("⚠️ No articles scraped")
            logger.info("Try running: python ingestion/upload_existing_data.py")
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
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        
        logger.info("=" * 50)
        logger.info("✅ Ingestion complete!")
        logger.info(f"📊 Total vectors in Pinecone: {total_vectors}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()