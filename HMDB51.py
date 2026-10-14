import requests
import os


def download_hmdb51(save_dir="./"):
    # 1. 파일 저장 경로 설정
    url = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
    file_name = "hmdb51_org.rar"
    save_path = os.path.join(save_dir, file_name)

    print(f"--- HMDB51 데이터셋 다운로드 시작 ---")
    print(f"저장 위치: {os.path.abspath(save_path)}")

    # 2. 스트리밍 방식으로 다운로드 (메모리 절약 및 안정성)
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 오류 발생 시 예외 처리

        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        chunk_size = 1024 * 1024  # 1MB 단위

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    # 진행률 표시
                    if total_size > 0:
                        percent = (downloaded_size / total_size) * 100
                        print(
                            f"\r진행률: {percent:.2f}% ({downloaded_size / (1024 ** 2):.1f}MB / {total_size / (1024 ** 2):.1f}MB)",
                            end="")

        print("\n\n--- 다운로드 완료! ---")
        print("이제 알집이나 7-Zip을 사용하여 수동으로 압축을 풀어주시면 됩니다.")

    except Exception as e:
        print(f"\n오류 발생: {e}")


if __name__ == "__main__":
    download_hmdb51()