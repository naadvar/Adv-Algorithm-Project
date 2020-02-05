from bs4 import BeautifulSoup
import requests
#here we scrape the given links and thenr return most relevant link
def clean_html(html_data):
    """
        Clean html
    """
    soup = BeautifulSoup(html_data, 'lxml')

    # remove all javascript and stylesheet code
    for ss_tag in soup(["script", "style"]):
        ss_tag.extract()

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())

    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text