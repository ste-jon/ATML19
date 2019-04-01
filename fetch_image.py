import urllib.request
from bs4 import BeautifulSoup
from PIL import Image
import requests
from io import BytesIO


def fetch_html(url):
    # contents = urllib.request.urlopen(url).read()
    contents = requests.get(url).content
    return contents


def find_img_src(html_doc):
    soup = BeautifulSoup(html_doc, 'html.parser')
    img_tag = soup.find("img", {"id": "imgBlkFront"})
    return img_tag.get("src")


def fetch_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save("data/covers/img.jpeg")


url = "https://www.amazon.com/dp/157120315X"
html = fetch_html(url)
img_src = find_img_src(html)
fetch_image(img_src)
