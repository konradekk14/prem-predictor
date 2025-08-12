import requests
from pathlib import Path

# data from last 4 seaosons
SEASONS = ['2122', '2223', '2324', '2425']
BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"

def download_season(season_code, out_path):
    # download a single season's data
    url = BASE_URL.format(season_code)
    print(f"Downloading {url} ...")
    r = requests.get(url)
    if r.status_code == 200:
        out_path.write_bytes(r.content)
        print(f"✓ Saved {out_path}")
        return True
    else:
        print(f"Failed to download {season_code}")
        return False

def ensure_all_seasons_downloaded(raw_dir):
    # download any missing season data
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print("Checking existing season data files...")
    missing_seasons = []
    
    for season in SEASONS:
        season_file = raw_dir / f"E0_{season}.csv"
        if season_file.exists():
            print(f"✓ Season {season} data already exists")
        else:
            print(f"⚠ Season {season} data missing - will download")
            missing_seasons.append(season)
    
    if missing_seasons:
        print(f"\nDownloading {len(missing_seasons)} missing season(s)...")
        success_count = 0
        for season in missing_seasons:
            out_file = raw_dir / f"E0_{season}.csv"
            if download_season(season, out_file):
                success_count += 1
        
        if success_count == len(missing_seasons):
            print(f"✓ All {success_count} missing seasons downloaded successfully")
        else:
            print(f"⚠ Only {success_count}/{len(missing_seasons)} seasons downloaded successfully")
    else:
        print("✓ All season data already available - no download needed")
    
    return len(missing_seasons) == 0

if __name__ == "__main__":
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    ensure_all_seasons_downloaded(raw_dir)