from requests_html import HTMLSession
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd


def get_page(urL):
    session = HTMLSession()
    r = session.get(urL)  # entry page
    if r.status_code == 200:
        print(f"Status code: {r.status_code}\n{urL} is successfully loaded!\n")
        return r.html
    else:
        print(f"Status code: {r.status_code}\n{urL} cannot be loaded!\n")
        return None


def append_comment(comments, comments_lists):
    for comment in comments:
        comment = " ".join(comment.text.split()) # .text is the functon of requests_html to extract text from a tag. .split() is to tokenize the sentence
        comments_lists.append(comment)


def main():
    movie_lists = []
    for x in range(2):
        r_html = get_page(f'https://www.metacritic.com/browse/movies/release-date/theaters/date?page={x}')
        if r_html is not None:
            movie_links = r_html.xpath("//a[@class = 'title']/@href")
            movie_lists += movie_links

    comments_lists = []
    for count, link in enumerate(movie_lists):
        r_html = get_page(f'https://www.metacritic.com{link}/user-reviews')

        if r_html is not None:

            if r_html.find("ul.pages", first=True) is None:
                if (r_html.xpath("//span[@class = 'blurb blurb_expanded']", first=True) is not None) or (
                        r_html.find("div.review_body > span:not([id]):not([class])", first=True) is not None):
                    comments = r_html.xpath("//span[@class = 'blurb blurb_expanded']")
                    append_comment(comments, comments_lists)

                    comments = r_html.find("div.review_body > span:not([id]):not([class])")
                    append_comment(comments, comments_lists)
            else:
                if (r_html.xpath("//span[@class = 'blurb blurb_expanded']", first=True) is not None) or (
                        r_html.find("div.review_body > span:not([id]):not([class])", first=True) is not None):

                    comments = r_html.xpath("//span[@class = 'blurb blurb_expanded']")
                    append_comment(comments, comments_lists)

                    comments = r_html.find("div.review_body > span:not([id]):not([class])")
                    append_comment(comments, comments_lists)

                    # Next page
                    for other_page_links in r_html.xpath('//a[@class = "page_num"]/@href'):
                        r_html = get_page(f'https://www.metacritic.com{other_page_links}')

                        if r_html is not None:
                            comments = r_html.xpath("//span[@class = 'blurb blurb_expanded']")
                            append_comment(comments, comments_lists)

                            comments = r_html.find("div.review_body > span:not([id]):not([class])")
                            append_comment(comments, comments_lists)

    # Return unique values in the list
    comments_lists = list(set(comments_lists))

    # Sentiment Analysis classifier
    classifier = pipeline("text-classification", model='distilbert-base-uncased-finetuned-sst-2-english')

    all_labeled_comments = []
    for comment in comments_lists:
        try:
            result = classifier(comment[:2376])
        except InvalidArgumentError:
            try:
                result = classifier(comment[:1900])
            except InvalidArgumentError:
                try:
                    result = classifier(comment[:1500])
                except InvalidArgumentError:
                    try:
                        result = classifier(comment[:1000])
                    except InvalidArgumentError:
                        try:
                            result = classifier(comment[:700])
                        except InvalidArgumentError:
                            result = classifier(comment[:600])
        scrap_result = {
            'Review': comment,
            'Sentiment': result[0]['label']
        }
        all_labeled_comments.append(scrap_result)

    df = pd.DataFrame(all_labeled_comments)
    print(df)

    df.to_csv(r'scrap_movie_reviews.csv')

    print("\nDONE SCRAPING!")


if __name__ == '__main__':
    main()
