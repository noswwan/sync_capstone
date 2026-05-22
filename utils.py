import whisper
import os
import fitz  # PyMuPDF

# 1. Whisper 모델 로드 (딱 한 번만 실행되도록 설정)
# 주의: main.py에서도 모델을 로드하고 있다면, 그쪽 코드는 지우고 이 model을 쓰는 게 좋아.
print("⏳ Whisper 모델을 로드 중입니다... (메모리 최적화 모드)")
model = whisper.load_model("base")

def get_transcript_via_whisper(file_path):
    """
    [면접 전용] 로컬 오디오 파일(.wav, .mp3 등)을 텍스트로 변환함.
    """
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None

    try:
        # 터미널에 찍히는 파일명이 'blob'이면 확장자가 없어 분석에 실패할 수 있어.
        # 이 함수는 들어온 파일을 최대한 분석하려고 시도해.
        print(f"🧠 목소리 분석 중... (파일명: {os.path.basename(file_path)})")
        
        # fp16=False는 CPU 환경에서 속도와 안정성을 위해 필수야 (GPU가 없다면)
        result = model.transcribe(file_path, fp16=False) 
        
        transcription = result.get('text', "").strip()
        if not transcription:
            print("⚠️ 음성 인식 결과가 비어 있습니다.")
            return None
            
        return transcription
    except Exception as e:
        print(f"❌ Whisper 처리 중 에러 발생: {e}")
        return None

def extract_text_from_pdf(file_path):
    """
    PDF 파일에서 텍스트를 추출함.
    """
    try:
        doc = fitz.open(file_path)
        text = "".join([page.get_text() for page in doc])
        doc.close()
        
        if not text.strip():
            print("⚠️ PDF에서 추출된 텍스트가 없습니다.")
            
        return text.strip()
    except Exception as e:
        print(f"❌ PDF 추출 중 에러 발생: {e}")
        return ""