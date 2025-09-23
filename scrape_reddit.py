import praw
import json
import os

# --- Step 1: Enter your Reddit API Credentials ---
# It's recommended to set these as environment variables for security.
# Or, you can replace the os.getenv('...') parts with your actual keys in strings.
# Example: client_id = "YOUR_CLIENT_ID_HERE"
CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', "VYCAvq__cji8zcGQvsMDhQ") 
CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', "Jro8WdYHp4CRL3xpDdbTRqULPB0vTQ")
USER_AGENT = os.getenv('REDDIT_USER_AGENT', "MyResearchScraper/1.0 by Capital_Crow_6646")

# --- Step 2: Configuration ---
SUBREDDIT_NAME = "eldercare"
POST_LIMIT = 10000
OUTPUT_FILENAME = "eldercare_posts.json"

def scrape_subreddit_posts(client_id, client_secret, user_agent, subreddit_name, limit):
    """
    Scrapes a specified number of recent posts from a subreddit and returns them
    in the desired format.
    """
    # Check if credentials are placeholders
    if "YOUR_CLIENT_ID_HERE" in client_id or "YOUR_CLIENT_SECRET_HERE" in client_secret:
        print("ERROR: Please replace the placeholder API credentials in the script.")
        return None

    print("Connecting to Reddit...")
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        # Verify that the connection is read-only and successful
        print(f"Successfully connected as {reddit.user.me()} (Read-Only Mode)")
    except Exception as e:
        print(f"Failed to connect to Reddit: {e}")
        return None
        
    subreddit = reddit.subreddit(subreddit_name)
    print(f"Fetching up to {limit} posts from r/{subreddit_name}. This may take a while...")

    scraped_data = {"stories": []}
    post_count = 0

    try:
        # Fetching the 'new' posts from the subreddit
        for submission in subreddit.new(limit=limit):
            post_object = {
                "title": submission.title,
                "story": submission.selftext
            }
            scraped_data["stories"].append(post_object)
            post_count += 1
            
            # Provide progress feedback in the terminal
            if post_count % 100 == 0:
                print(f"  ...processed {post_count} posts...")

    except Exception as e:
        print(f"An error occurred while fetching posts: {e}")
        print("Saving the data collected so far.")

    print(f"\nFinished scraping. Total posts collected: {post_count}")
    return scraped_data

def save_to_json(data, filename):
    """Saves the collected data to a JSON file."""
    if data is None or not data["stories"]:
        print("No data was collected, skipping file save.")
        return
        
    print(f"Saving data to {filename}...")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Successfully saved data to {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")

# --- Main execution ---
if __name__ == "__main__":
    collected_posts = scrape_subreddit_posts(
        CLIENT_ID, 
        CLIENT_SECRET, 
        USER_AGENT, 
        SUBREDDIT_NAME, 
        POST_LIMIT
    )
    save_to_json(collected_posts, OUTPUT_FILENAME)