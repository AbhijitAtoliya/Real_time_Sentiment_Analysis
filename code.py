import pandas as pd
import httpx
import asyncio
from bs4 import BeautifulSoup
import nest_asyncio
import logging
from textblob import TextBlob
import matplotlib.pyplot as plt

# Apply necessary asyncio adjustments, which might be necessary if running in an environment similar to Jupyter.
nest_asyncio.apply()

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

def perform_google_search(query, max_results=100):
    from googlesearch import search
    """Perform a Google search and collect up to max_results URLs, focusing on articles only."""
    results = []
    try:
        search_query = f"{query} -site:youtube.com -site:vimeo.com"
        for result in search(search_query):
            results.append(result)
            if len(results) >= max_results:
                break
    except Exception as e:
        logging.error("An error occurred while fetching the results: " + str(e))
    return results

async def fetch_article_content(url):
    """Asynchronously fetch content from a URL using httpx with a user-agent."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers=headers) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.RequestError as e:
        logging.error(f"Request error for {url}: {str(e)}")
        return None
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP status error for {url}: {e.response.status_code} - {e.response.reason_phrase}")
        return None

async def process_articles(urls):
    """Process a list of URLs to fetch their content."""
    contents = []
    async with httpx.AsyncClient() as client:
        tasks = [fetch_article_content(url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for response in responses:
            if response:
                soup = BeautifulSoup(response, 'html.parser')
                text = ' '.join(p.get_text(strip=True) for p in soup.find_all('p'))
                if text:
                    contents.append(text)
                else:
                    contents.append("No content available")
            else:
                contents.append("No content available")
    return contents

async def add_content_to_df(query):
    """Main function to manage the workflow from search to DataFrame creation."""
    urls = perform_google_search(query, max_results=100)
    contents = await process_articles(urls)
    df = pd.DataFrame(contents, columns=['content'])
    df = df[df['content'].str.strip() != '']
    return df

async def main():
    user_query = input("Please enter your search query: ")
    df = await add_content_to_df(user_query)

    def classify_sentiment(text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment'] = df['content'].apply(classify_sentiment)
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_percentages = sentiment_counts / len(df) * 100

    plt.figure(figsize=(4, 4))
    plt.pie(sentiment_percentages, labels=sentiment_percentages.index, autopct='%1.1f%%', startangle=140, colors=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#FFD700'])
    plt.title('Sentiment Distribution in Articles')
    plt.axis('equal')
    plt.show()

    overall_sentiment = sentiment_percentages.idxmax()
    print(f"The overall sentiment is {overall_sentiment}.")

if __name__ == "__main__":
    asyncio.run(main())
