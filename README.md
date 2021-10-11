# Movie Reviews Sentiment Analysis
This is a **movie sentiment analysis** project that we will use the 
[IMDB](https://ai.stanford.edu/~amaas/data/sentiment/) dataset to train our machine learning models. Then, we use these 
models to classify whether a movie review is positive or negative.

## Authors

- Lim Chia Chung [@Jacky0111](https://www.github.com/Jacky0111)
- Lim Ming Jun [@mingjun1120](https://www.github.com/mingjun1120)
- Leong Yit Wee [@leongyitwee](https://github.com/leongyitwee)

## Python Version
Python 3.8

## Application Used
Google Colab

## Documentation
- Model Training (DistilBert): [Ktrain](https://github.com/amaiya/ktrain)
- Crawling Data: [request-html](https://docs.python-requests.org/projects/requests-html/en/latest/)

## How to run
Open the `NLP Assignment` folder in `Google Colab`. Then, follow these steps:

1. Run the `2_preprocess_data.ipynb`
2. Run `DistilBERT.ipynb` inside **Lim Ming Jun** folder
3. Run `Long_Short-Term_Memory(LSTM).ipynb` inside **Lim Chia Chung** folder
4. Run `Convolutional Neural Network Tutorial (CNN).ipynb` inside **Leong Yit Wee** folder
5. Run `Local Sentiment Analysis App.ipynb`

## Web Scraping
For this part, go to the **MovieScraper** folder first and the run the `main.py` (scraper). The output will 
store as `scrap_movie_reviews.csv`.

## How this program works:
Basically, we use the [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/) dataset
to train on the **DistilBert**, **LSTM** and **CNN** models. Then, we will compare the performance of these 3 models
via `Local Sentiment Analysis App.ipynb` where this notebook can do a **single review analysis** input by users as well as 
**multiple reviews analysis** by feeding the data that we have scraped through `main.py` in **MovieScraper** folder.

#### Note:
The scraped data are well labeled with **`POSITIVE`** and **`NEGATIVE`** sentiment.
