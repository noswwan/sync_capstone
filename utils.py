import yt_dlp
import whisper
import os
import re
import fitz

def extract_text_from_pdf(file_path):
    """
    PDF 파일의 경로를 받아 텍스트를 추출해 반환한다.
    """
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"PDF 추출 중 에러 발생: {e}")
        return ""

# 1. Whisper 모델 로드
# base: 속도가 빠름 / small: 정확도가 더 높음 (현재는 속도를 위해 base 사용)
print("⏳ Whisper 모델을 로드 중입니다... (최초 실행 시 시간이 소요될 수 있음)")
model = whisper.load_model("base")

def extract_video_id(url):
    """
    유튜브 URL에서 11자리의 비디오 ID를 추출합니다.
    """
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript_via_whisper(url):
    """
    [핵심 로직]
    1. 유튜브에서 오디오 추출 (yt-dlp)
    2. Whisper AI로 받아쓰기 수행 (STT)
    3. 작업 완료 후 임시 오디오 파일 삭제
    """
    video_id = extract_video_id(url)
    if not video_id:
        print("❌ 유효하지 않은 유튜브 URL입니다.")
        return None

    # 임시로 저장될 오디오 파일명
    audio_filename = f"temp_{video_id}.mp3"

    # 2. yt-dlp 설정 (오디오 추출 옵션)
    ydl_opts = {
        'format': 'bestaudio/best',
        'ffmpeg_location': './',  # ✅ 프로젝트 루트 폴더에 ffmpeg.exe가 있어야 함
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f"temp_{video_id}", # 확장자는 postprocessor가 mp3로 붙여줌
        'quiet': True,                 # 로그 너무 많이 찍히지 않게 설정
    }

    try:
        # Step A: 유튜브 오디오 다운로드
        print(f"🎵 오디오 데이터 추출 중... (ID: {video_id})")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Step B: Whisper AI 전사 (Speech-to-Text)
        # 파일이 실제로 생성될 때까지 약간의 대기 시간이 필요할 수 있음
        if not os.path.exists(audio_filename):
            # 일부 환경에서 확장자 없이 생성되는 경우 대응
            if os.path.exists(f"temp_{video_id}"):
                os.rename(f"temp_{video_id}", audio_filename)

        print("🧠 AI가 영상을 분석하여 텍스트로 변환 중입니다...")
        result = model.transcribe(audio_filename)
        full_text = result['text']

        # Step C: 로컬 용량 관리를 위해 임시 오디오 파일 삭제
        if os.path.exists(audio_filename):
            os.remove(audio_filename)

        print("✅ 전사 완료!")
        return full_text

    except Exception as e:
        print(f"❌ Whisper 처리 중 에러 발생: {e}")
        if os.path.exists(audio_filename):
            os.remove(audio_filename)
        return None
    
