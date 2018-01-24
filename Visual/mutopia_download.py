#%% Beethoven Piano Sonatas
from bs4 import BeautifulSoup
import urllib.request
import re

url = ['http://www.mutopiaproject.org/cgibin/make-table.cgi?startat=1&searchingfor=&Composer=&Instrument=Piano&Style=&collection=&id=&solo=&recent=&timelength=&timeunit=&lilyversion=&preview=']

html_page = urllib.request.urlopen(url) 
soup = BeautifulSoup(html_page)

for link in soup.findAll('a', attrs={'href': re.compile("^http://")}):
    print(link.get('href'))
