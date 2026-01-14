"""
Debug script to inspect insightfulsermons.com structure.
This will help us figure out what selectors to use.

Usage:
    python ingestion/debug_scraper.py
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

def debug_site():
    """Inspect the site and print what we find"""
    
    # Setup Chrome
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Comment out to see browser
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print("=" * 60)
        print("Debugging insightfulsermons.com")
        print("=" * 60)
        
        url = "https://www.insightfulsermons.com/"
        print(f"\n1. Loading: {url}")
        driver.get(url)
        time.sleep(3)  # Wait for page to load
        
        print(f"   ✅ Page title: {driver.title}")
        
        # Check for various possible selectors
        print("\n2. Looking for category links...")
        
        selectors_to_try = [
            ".blog-category-list a.blog-link",
            ".blog-category-list a",
            "a.blog-link",
            ".category-link",
            "[class*='category'] a",
            "[class*='blog'] a",
            "nav a",
            ".nav a",
        ]
        
        for selector in selectors_to_try:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(f"   ✅ Found {len(elements)} elements with: {selector}")
                    for i, elem in enumerate(elements[:3]):  # Show first 3
                        try:
                            text = elem.text.strip()
                            href = elem.get_attribute('href')
                            print(f"      [{i}] Text: '{text}' | URL: {href}")
                        except:
                            pass
                else:
                    print(f"   ❌ No elements with: {selector}")
            except Exception as e:
                print(f"   ❌ Error with {selector}: {e}")
        
        # Check for blog posts
        print("\n3. Looking for blog posts...")
        
        post_selectors = [
            ".blog-post",
            "[class*='blog-post']",
            "[class*='post']",
            "article",
            ".article",
        ]
        
        for selector in post_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(f"   ✅ Found {len(elements)} posts with: {selector}")
                    
                    # Try to find title and content in first post
                    if elements:
                        post = elements[0]
                        print(f"\n   Inspecting first post:")
                        
                        # Look for title
                        title_selectors = [
                            ".blog-title-link",
                            ".blog-link",
                            "h1", "h2", "h3",
                            "[class*='title']",
                            "a"
                        ]
                        
                        for ts in title_selectors:
                            try:
                                title_elem = post.find_element(By.CSS_SELECTOR, ts)
                                if title_elem:
                                    print(f"      Title selector '{ts}': {title_elem.text[:50]}")
                                    break
                            except:
                                pass
                        
                        # Look for content
                        content_selectors = [
                            ".paragraph",
                            "p",
                            "[class*='content']",
                            "div"
                        ]
                        
                        for cs in content_selectors:
                            try:
                                content_elems = post.find_elements(By.CSS_SELECTOR, cs)
                                if content_elems:
                                    print(f"      Content selector '{cs}': Found {len(content_elems)} elements")
                                    break
                            except:
                                pass
                    
                else:
                    print(f"   ❌ No posts with: {selector}")
            except Exception as e:
                print(f"   ❌ Error with {selector}: {e}")
        
        # Save page source for manual inspection
        print("\n4. Saving page source...")
        with open('debug_page_source.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        print("   ✅ Saved to: debug_page_source.html")
        
        # Take screenshot
        print("\n5. Taking screenshot...")
        driver.save_screenshot('debug_screenshot.png')
        print("   ✅ Saved to: debug_screenshot.png")
        
        print("\n" + "=" * 60)
        print("✅ Debug complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Check debug_page_source.html in a text editor")
        print("  2. Look at debug_screenshot.png")
        print("  3. Share findings so I can update the scraper")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.quit()

if __name__ == "__main__":
    debug_site()