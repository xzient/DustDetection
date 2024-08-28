# prompt: get today's hour and minutes


from urllib.parse import urlparse
import datetime
import requests
from pathlib import Path
from bs4 import BeautifulSoup

# !pip install beautifulsoup4 requests


def get_main_page(url):
  parsed_url = urlparse(url)
  return f"{parsed_url.scheme}://{parsed_url.netloc}"

def get_today_str():
  today = datetime.date.today()
  now = datetime.datetime.now()
  hour = now.hour
  minute = now.minute
  return today.strftime("%Y-%m-%d"), hour, minute

def scrape_image(url, filename, num):
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')
  img_tag = soup.find_all('img')[num] # Adjust this based on the website's structure
  img_url = img_tag['src']
  main_page = get_main_page(url)
  img_response = requests.get(main_page + img_url)
  with open(filename, 'wb') as f:
    f.write(img_response.content)
  print(f"Image saved as {filename}")

def get_images(directory, url=None):
  url = "https://www.wyoroad.info/highway/webcameras/view?site=I80PineBluffs" if url == None else url  # Replace with the actual website URL
  
  Path( "images/" + directory).mkdir(parents=True, exist_ok=True)
  
  filename_base = "Wyoming"  
  date, hour, minute = get_today_str()
  date_prefix = date + "_" + str(hour) + "_" + str(minute)

  drop_data = "images/" + directory 

  for i in range(2,5):
    
    scrape_image(url, drop_data + filename_base + "_" + date_prefix + "_" + str(i) + ".jpg", i)
