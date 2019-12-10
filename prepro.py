import glob
import requests
import datetime
from tqdm import tqdm

tsv_paths = glob.glob('./datasets/*.tsv')
for path in tsv_paths:
    with open(path, 'r') as f:
        f.readline()
        for url in tqdm(f.readlines()):
            url = url.split('\t')[0]
            print(url)
            ext = url.split('.')[-1]
            save_path = './datasets/' + datetime.datetime.now().strftime('%Y%m%d_%H%M-%S_%f' + '.' + ext)
            response = requests.get(url)

            with open(save_path, 'wb') as wf:
                wf.write(response.content)
