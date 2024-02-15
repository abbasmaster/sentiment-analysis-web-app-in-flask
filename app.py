from flask import Flask, render_template, request, jsonify
import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup


app = Flask(__name__)

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt', max_length=512, truncation=True)
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        website_choice = request.form.get('website_choice')
        url = request.form['url']
        if website_choice == 'yelp':
            reviews = scrape_yelp_reviews(url)
        elif website_choice == 'imdb':
            reviews = scrape_imdb_reviews(url)
        else:
            return "Invalid website choice"

        if reviews:
            df = analyze_sentiment(reviews)
            return render_template('index.html', df=df.to_html(classes='table table-striped'), error=None)
        else:
            error = "No reviews found on the webpage."
            return render_template('index.html', df=None, error=error)
    return render_template('index.html', df=None, error=None)

# Define a function to scrape and analyze Yelp reviews
def scrape_yelp_reviews(url):
    try:
        r = requests.get(url)
        r.raise_for_status()  # Raise an exception if the request fails
        soup = BeautifulSoup(r.text, 'html.parser')
        regex = re.compile('.*comment.*')
        results = soup.find_all('p', {'class': regex})
        reviews = [result.text for result in results]
        return reviews
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Yelp: {e}")
        return []

# Define a function to scrape and analyze IMDb reviews (change regex as needed)
def scrape_imdb_reviews(url):
    try:
        # Check if the URL contains "reviews" and add it if not
        if "reviews" not in url:
            url += "reviews"
        
        r = requests.get(url)
        r.raise_for_status()  # Raise an exception if the request fails
        soup = BeautifulSoup(r.text, 'html.parser')
        regex = re.compile('.*(text show-more__control clickable|text show-more__control).*')
        results = soup.find_all('div', {'class': regex})
        reviews = [result.text for result in results]
        return reviews
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from IMDb: {e}")
        return []

def analyze_sentiment(reviews):
    df = pd.DataFrame(np.array(reviews), columns=['review'])
    df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x))
    return df

if __name__ == '__main__':
    app.run(debug=True)