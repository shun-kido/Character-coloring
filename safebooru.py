import urllib.error
import urllib.request
from bs4 import BeautifulSoup
import time
import os

def download_image(url, dst_path):
    try:
        data = urllib.request.urlopen(url).read()
        with open(dst_path, mode="wb") as f:
            f.write(data)
    except urllib.error.HTTPError as e:
        url = url.replace('samples', 'images').replace('sample_', '')
        print(url)
        data = urllib.request.urlopen(url).read()
        with open(dst_path, mode="wb") as f:
            f.write(data)

#range内がDLページ数、40/1page
for i in range(5):
    #white_backgroundタグの画像を抽出
    url = 'https://safebooru.org/index.php?page=post&s=list&tags=white_background&pid='+str(i*40)
    ua = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) '\
         'AppleWebKit/537.36 (KHTML, like Gecko) '\
         'Chrome/55.0.2883.95 Safari/537.36 '

    #req = urllib.request.Request(url, headers={'User-Agent': ua})
    html = urllib.request.urlopen(url)

    soup = BeautifulSoup(html, "html.parser")

    img_list = soup.find(class_="content").find_all('img')
    url_list = []
    for img in img_list:
        url_list.append(img.get('src'))

    download_dir = './safebooru'
    sleep_time_sec = 1

    for i, url in enumerate(url_list):
        url = url.split('?')
        url[0] = url[0].replace('thumbnail', 'sample')
        filename = (url[1].lstrip('?'))
        dst_path = os.path.join(download_dir, filename+'.jpg')
        time.sleep(sleep_time_sec)
        print(url, dst_path)
        download_image('https:'+url[0], dst_path)
            