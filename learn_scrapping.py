import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
#importer le code de la page 
url='https://www.avito.ma/fr/maroc/voitures_d_occasion-%C3%A0_vendre'
reponse = requests.get(url)
soup=BeautifulSoup(reponse.text, 'html.parser')
print(soup) 