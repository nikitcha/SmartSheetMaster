#%% Beethoven Piano Sonatas
from bs4 import BeautifulSoup
import urllib.request
import re

url = ['http://www.mutopiaproject.org/cgibin/make-table.cgi?collection=beetson',
       'http://www.mutopiaproject.org/cgibin/make-table.cgi?startat=10&searchingfor=&Composer=&Instrument=&Style=&collection=beetson&id=&solo=&recent=&timelength=&timeunit=&lilyversion=&preview=',
       'http://www.mutopiaproject.org/cgibin/make-table.cgi?startat=20&searchingfor=&Composer=&Instrument=&Style=&collection=beetson&id=&solo=&recent=&timelength=&timeunit=&lilyversion=&preview=']


html_page = urllib.request.urlopen(url) 
soup = BeautifulSoup(html_page)

for link in soup.findAll('a', attrs={'href': re.compile("^http://")}):
    print(link.get('href'))
