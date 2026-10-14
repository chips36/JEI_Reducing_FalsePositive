"""Collect WWE broadcast videos for the male-chest video evaluation (Sec. 4.3.1)."""
import os
import sys

try:
    from yt_dlp import YoutubeDL
except ImportError:
    print("yt-dlp가 설치되지 않았습니다. 'pip install yt-dlp'를 실행해 주세요.")
    sys.exit(1)

# 1. Output directory
SAVE_DIR = "wwe_research_data"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 2. Search keywords or channel URLs to collect
# Use the 'ytsearchN:' prefix to collect the top N search results.
search_queries = [
    "ytsearch50:WWE Full Match 1080p",  # Top 50 search results
    "ytsearch50:WWE Highlights shirtless",  # Top 50 search results
    "https://www.youtube.com/@WWE/videos"  # Latest videos from the official WWE channel
]

ydl_opts = {
    # Video quality: best available between 720p and 1080p
    'format': 'bestvideo[height<=1080][height>=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best',
    'outtmpl': f'{SAVE_DIR}/%(title)s.%(ext)s',

    'nocheckcertificate': True,
    'ignoreerrors': True,  # Skip unavailable videos instead of stopping
    'quiet': False,
    'no_warnings': False,
    # Player clients used by the YouTube extractor
    'extractor_args': {'youtube': {'player_client': ['android', 'web']}},

    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    # Random delay between downloads (seconds)
    'sleep_interval': 3,
    'max_sleep_interval': 7,
}

def download_wwe_data():
    """Download videos for every query in search_queries."""
    print(f"--- 대표님, WWE 고화질 데이터 수집을 시작합니다 ---")
    with YoutubeDL(ydl_opts) as ydl:
        for query in search_queries:
            print(f"\n[대상]: {query}")
            try:
                ydl.download([query])
            except Exception as e:
                print(f"다운로드 중 오류 발생: {e}")


if __name__ == "__main__":
    download_wwe_data()
