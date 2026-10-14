import os
import sys

try:
    from yt_dlp import YoutubeDL
except ImportError:
    print("yt-dlp가 설치되지 않았습니다. 'pip install yt-dlp'를 실행해 주세요.")
    sys.exit(1)

# 1. 저장 경로 설정
SAVE_DIR = "wwe_research_data"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 2. 수집할 키워드 또는 채널 URL
# 검색 결과 전체를 수집하려면 'ytsearch' 키워드를 사용합니다.
search_queries = [
    "ytsearch50:WWE Full Match 1080p",  # 상위 50개 검색 결과
    "ytsearch50:WWE Highlights shirtless",  # 상위 50개 검색 결과
    "https://www.youtube.com/@WWE/videos"  # WWE 공식 채널 최신 영상들
]

ydl_opts = {
    # 화질 설정: 720p~1080p 사이 최적 선택
    'format': 'bestvideo[height<=1080][height>=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best',
    'outtmpl': f'{SAVE_DIR}/%(title)s.%(ext)s',

    # [핵심] 유튜브 차단 우회용 인자 추가
    'nocheckcertificate': True,
    'ignoreerrors': True,
    'quiet': False,
    'no_warnings': False,
    # 유튜브의 '봇 감지'를 피하기 위한 클라이언트 위장
    'extractor_args': {'youtube': {'player_client': ['android', 'web']}},

    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'sleep_interval': 3,
    'max_sleep_interval': 7,
}

def download_wwe_data():
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