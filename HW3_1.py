import gc
from urllib.request import urlretrieve
import gzip
import shutil
import os
import pandas as pd

# download and extract data
if not os.path.exists('CD_and_Vinyl.json.gz'):
    print('downloading data')
    url = 'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFiles/Books.json.gz'
    dest = 'CD_and_Vinyl.json.gz'
    urlretrieve(url, dest)
    if not os.path.exists('CD_and_Vinyl.json'):
        with gzip.open('CD_and_Vinyl.json.gz', 'rb') as f_in:
            with open('CD_and_Vinyl.json', 'wb') as f_out:
                print('extracting data')
                shutil.copyfileobj(f_in, f_out)

df = pd.read_json('CD_and_Vinyl.json')
df.head()
