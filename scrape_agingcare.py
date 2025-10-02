import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re

# Configuration
BASE_URL = "https://www.agingcare.com"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
DELAY_BETWEEN_QUESTIONS = 0.5  # seconds
DELAY_BETWEEN_PAGES = 2    # seconds
MAX_QUESTIONS_PER_TOPIC = 20
OUTPUT_DIR = 'AgingCare_Posts'
TOPIC_LINKS_FILE = 'agingcare_topic_links.json'

def get_max_pages(session, topic_url):
    """Fetch the first page of a topic and determine the total number of pages."""
    try:
        response = session.get(BASE_URL + topic_url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        max_page = 1
        pag_links = soup.find_all('a', href=re.compile(r'page=\d+'))
        for link in pag_links:
            match = re.search(r'page=(\d+)', link.get('href', ''))
            if match:
                max_page = max(max_page, int(match.group(1)))
        
        return max_page
    except Exception as e:
        print(f"Error detecting max pages for {topic_url}: {e}")
        return 1

def extract_question_links(soup, remaining_needed):
    """Extract up to `remaining_needed` question titles and URLs from a list page."""
    questions = []
    ul = soup.find('ul', class_='unstyled', attrs={'data-page': 'true'})
    if not ul:
        print("Warning: Could not find question list <ul>.")
        return questions
    
    for li in ul.find_all('li', class_='item'):
        if len(questions) >= remaining_needed:
            break
        a_tag = li.find('a', class_='blue-link')
        if a_tag and a_tag.find('h3'):
            title = a_tag.find('h3').get_text(strip=True)
            if title:
                url = BASE_URL + a_tag['href']
                questions.append({'title': title, 'url': url})
    return questions

def extract_question_content(session, url, title):
    """Fetch a question page and extract the original post body."""
    try:
        response = session.get(url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        section = soup.find('section', class_='Content Question-Content')
        if not section:
            print(f"Warning: Could not find <section class='Content Question-Content'> for '{title[:50]}...'")
            return ""
        
        content_div = section.find('div', class_='text bodyNeutral black pad-btm-l', attrs={'itemprop': 'text'})
        if not content_div:
            print(f"Warning: Could not find content <div> for '{title[:50]}...'")
            return ""
        
        paragraphs = content_div.find_all('p')
        content_parts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
        content = ' '.join(content_parts) if content_parts else ""
        
        if not content:
            print(f"Warning: No content extracted for '{title[:50]}...'")
        
        return content
    except Exception as e:
        print(f"Error fetching/extracting {url}: {e}")
        return ""

def sanitize_filename(name):
    """Convert topic name to a valid filename."""
    # Remove invalid characters and replace spaces with underscores
    return re.sub(r'[^\w\s-]', '', name.replace(' Questions', '')).strip().replace(' ', '_') + '.json'

def scrape_topic(session, topic_name, topic_url):
    """Scrape up to MAX_QUESTIONS_PER_TOPIC questions for a single topic."""
    stories = []
    page_num = 1
    max_pages = get_max_pages(session, topic_url)
    print(f"Scraping topic '{topic_name}' ({max_pages} pages)...")
    
    while len(stories) < MAX_QUESTIONS_PER_TOPIC and page_num <= max_pages:
        page_url = f"{BASE_URL}{topic_url}?page={page_num}" if page_num > 1 else f"{BASE_URL}{topic_url}"
        print(f"  Page {page_num}/{max_pages}...")
        
        try:
            response = session.get(page_url, headers=HEADERS)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            remaining_needed = MAX_QUESTIONS_PER_TOPIC - len(stories)
            page_questions = extract_question_links(soup, remaining_needed)
            print(f"  Found {len(page_questions)} questions on page {page_num}")
            
            if not page_questions:
                print("  No more questions found. Stopping this topic.")
                break
            
            for q in page_questions:
                content = extract_question_content(session, q['url'], q['title'])
                stories.append({
                    'title': q['title'],
                    'story': content.strip()
                })
                print(f"  Scraped: {q['title'][:50]}... ({len(content)} chars)")
                time.sleep(DELAY_BETWEEN_QUESTIONS)
                
                if len(stories) >= MAX_QUESTIONS_PER_TOPIC:
                    break
            
            page_num += 1
            time.sleep(DELAY_BETWEEN_PAGES)
        except Exception as e:
            print(f"  Error scraping page {page_num} for {topic_name}: {e}")
            break
    
    return stories

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read topic links
    try:
        with open(TOPIC_LINKS_FILE, 'r', encoding='utf-8') as f:
            topic_links = json.load(f)
    except Exception as e:
        print(f"Error reading {TOPIC_LINKS_FILE}: {e}")
        return
    
    session = requests.Session()
    
    for topic in topic_links:
        topic_name = topic['topic']
        topic_url = topic['link'].replace(BASE_URL, '')  # Get relative URL
        stories = scrape_topic(session, topic_name, topic_url)
        
        # Save to individual JSON file
        output_file = os.path.join(OUTPUT_DIR, sanitize_filename(topic_name))
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({'stories': stories}, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(stories)} stories to {output_file}\n")

if __name__ == "__main__":
    main()