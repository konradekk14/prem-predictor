# download_data.py
import requests
from pathlib import Path

seasons = ['2122','2223','2324','2425']  # last 3-4 seasons
out_dir = Path('data')
out_dir.mkdir(exist_ok=True)

for s in seasons:
    url = f'https://www.football-data.co.uk/mmz4281/{s}/E0.csv'
    r = requests.get(url)
    if r.status_code == 200:
        (out_dir / f'E0_{s}.csv').write_bytes(r.content)
    else:
        print('missing', url)
