import requests
from bs4 import BeautifulSoup
import json

# Configuration
BASE_URL = "https://www.agingcare.com"
TOPICS_URL = "/topics"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
OUTPUT_FILE = 'agingcare_topic_links.json'

def extract_topic_links(soup):
    """Extract topic names and their question page links."""
    topics = []
    # Find all <a> tags with the specified class and href pattern
    topic_links = soup.find_all('a', class_='text small blue-link semi-bold fancy-container', href=lambda h: h and h.startswith('/topics/') and h.endswith('/questions'))
    
    for a_tag in topic_links:
        topic_name = a_tag.get_text(strip=True)
        if topic_name:
            # Remove the count (e.g., "(910)") from the topic name
            topic_name = topic_name.split('\xa0')[0].strip()
            link = BASE_URL + a_tag['href']
            topics.append({
                'topic': topic_name,
                'link': link
            })
    
    return topics

def main():
    session = requests.Session()
    
    print(f"Scraping topic links from {BASE_URL}{TOPICS_URL}...")
    try:
        response = session.get(BASE_URL + TOPICS_URL, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        topic_links = extract_topic_links(soup)
        print(f"Found {len(topic_links)} topic links")
        
        # Save to JSON
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(topic_links, f, indent=4, ensure_ascii=False)
        
        print(f"Saved {len(topic_links)} topic links to {OUTPUT_FILE}")
        
        # Print topics for verification
        for topic in topic_links:
            print(f"Topic: {topic['topic']} -> {topic['link']}")
            
    except Exception as e:
        print(f"Error scraping topics page: {e}")

if __name__ == "__main__":
    main()