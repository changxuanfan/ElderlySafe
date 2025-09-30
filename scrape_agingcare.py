import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import requests  # For any static fallback, but mainly Selenium here

# Configuration
DELAY = 2  # Seconds between requests
OUTPUT_FILE = 'agingcare_discussions.json'
MAIN_URL = 'https://www.agingcare.com/topics'
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize driver (assumes ChromeDriver in PATH)
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Run without GUI; remove for debugging
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

discussions = []

def extract_plain_text(html_content):
    """Extract plain text from HTML, stripping tags and normalizing."""
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove script/style, get text, clean up
    for script in soup(["script", "style", "nav", "footer"]):
        script.decompose()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    return text

try:
    driver.get(MAIN_URL)
    time.sleep(DELAY)

    # Navigate A-Z letters (assuming links are <a> with href containing 'letter=' or similar; adjust if needed)
    for letter in LETTERS:
        try:
            # Find and click letter link (inspect site: likely class='letter-link' or href=f'javascript:loadLetter("{letter}")')
            # Fallback: construct URL if direct, e.g., MAIN_URL + f'?letter={letter.lower()}'
            letter_url = f'{MAIN_URL}?letter={letter.lower()}'
            driver.get(letter_url)
            time.sleep(DELAY)

            # Extract topic links on letter page (e.g., <a class="topic-link" href="/topics/ID/name">)
            topic_links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/topics/"]')  # Adjust selector as needed
            for link in topic_links:
                topic_href = link.get_attribute('href')
                if '/topics/' in topic_href and topic_href not in [t['url'] for t in discussions]:  # Avoid dups
                    topic_name = link.text.strip() or topic_href.split('/')[-1].replace('-', ' ').title()
                    print(f"Processing topic: {topic_name}")

                    # Go to topic page
                    driver.get(topic_href)
                    time.sleep(DELAY)

                    # Click "Discussions" tab/section (e.g., <a href="#discussions"> or class="tab-discussions")
                    try:
                        discussions_tab = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-tab="discussions"], a[href*="discussions"], .discussions-link')))
                        discussions_tab.click()
                        time.sleep(DELAY)
                    except TimeoutException:
                        print(f"No discussions tab found for {topic_name}; skipping.")
                        continue

                    # Extract discussion list (paginated)
                    page_num = 1
                    while True:
                        # Find discussion items (e.g., <div class="discussion-item"> with <a> title and href)
                        disc_elements = driver.find_elements(By.CSS_SELECTOR, '.discussion-item, .forum-thread, a[href*="/questions/"], a[href*="/discussions/"]')
                        new_discs = False
                        for elem in disc_elements:
                            try:
                                disc_title_elem = elem.find_element(By.CSS_SELECTOR, 'a, h3, .title')
                                disc_title = disc_title_elem.text.strip()
                                disc_url = disc_title_elem.get_attribute('href')
                                if not disc_url or disc_title in [d['title'] for d in discussions]:
                                    continue
                                new_discs = True

                                # Go to discussion thread
                                driver.get(disc_url)
                                time.sleep(DELAY)

                                # Extract title and full content
                                thread_title = driver.find_element(By.CSS_SELECTOR, 'h1, .thread-title').text.strip()
                                thread_html = driver.find_element(By.CSS_SELECTOR, '.post-content, .thread-body, main').get_attribute('innerHTML')
                                plain_text = extract_plain_text(thread_html)

                                discussions.append({
                                    "topic": topic_name,
                                    "title": thread_title,
                                    "plain_text": plain_text
                                })
                                print(f"Scraped discussion: {thread_title[:50]}...")

                                # Back to discussions list
                                driver.get(topic_href)
                                discussions_tab.click()
                                time.sleep(DELAY)
                            except NoSuchElementException:
                                continue

                        if not new_discs:
                            break

                        # Check for next page (e.g., <a class="next">)
                        try:
                            next_btn = driver.find_element(By.CSS_SELECTOR, '.next-page, a[rel="next"]')
                            if 'disabled' in next_btn.get_attribute('class') or not next_btn.is_enabled():
                                break
                            next_btn.click()
                            time.sleep(DELAY)
                            page_num += 1
                        except NoSuchElementException:
                            break

        except Exception as e:
            print(f"Error processing letter {letter}: {e}")
            continue

finally:
    driver.quit()

# Save to JSON
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(discussions, f, ensure_ascii=False, indent=2)

print(f"Scraping complete! {len(discussions)} discussions saved to {OUTPUT_FILE}.")