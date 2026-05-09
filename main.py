import os
import json
import asyncio
import re
import shutil
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, Optional
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# 외부 라이브러리
import azure.cognitiveservices.speech as speechsdk
import google.generativeai as genai

# 내부 모듈
from utils import get_transcript_via_whisper, extract_text_from_pdf
from analyzer import get_total_weak_patterns
from interview_manager import InterviewManager


# 1. 환경 변수 로드 및 AI 설정
load_dotenv()
AZURE_KEY = os.getenv('AZURE_SPEECH_KEY')
AZURE_REGION = os.getenv('AZURE_SPEECH_REGION')
GEMINI_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('models/gemini-3.1-flash-lite-preview')
interview_manager = InterviewManager(gemini_model)

app = FastAPI(title="Sync Capstone API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists("static/audio"):
    os.makedirs("static/audio")
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- [ 데이터 모델 정의 ] ---

class InterviewSetup(BaseModel):
    position: str
    tech_stack: List[str]
    experience_level: str
    project_summary: str
    interview_mode: str

class AnswerRequest(BaseModel):
    current_question: str
    user_answer: str


# --- [ 내부 로직 함수군 ] ---

def process_pronunciation_result(result_obj):
    """[Azure 발음 분석 상세 정제]"""
    actual_result = getattr(result_obj, "result", result_obj)
    pron_result = speechsdk.PronunciationAssessmentResult(actual_result)
    
    word_details = []
    for word in pron_result.words:
        phonemes = [{"ph": ph.phoneme, "score": ph.accuracy_score} for ph in word.phonemes]
        word_details.append({
            "word": word.word,
            "accuracy": word.accuracy_score,
            "phonemes": phonemes
        })

    return {
        "sentence": {
            "text": actual_result.text,
            "accuracy": pron_result.accuracy_score,
            "pronunciation": pron_result.pronunciation_score,
            "fluency": pron_result.fluency_score
        },
        "words": word_details
    }

async def get_ai_coaching(final_data, current_question):
    """[Gemini 3.1 Flash Lite] 질문-대답 일치성 및 발음 코칭"""
    prompt = f"""
    당신은 1:1 영어 회화 튜터입니다. 사용자의 답변 내용을 분석해 주세요.

    [상황]
    - 선생님의 질문: {current_question}
    - 학생의 답변: {final_data['sentence']['text']}
    - 발음 데이터: {final_data['words']}

    [분석 요청]
    1. 대답 적절성: 학생이 질문의 의도에 맞는 대답을 했는지 한국어로 다정하게 평가해 주세요.
    2. 발음 코칭: 음소(phonemes) 점수가 낮은 단어를 콕 집어 발음 팁(입모양 등)을 주세요.
    3. 모범 답안: 'Model Answer: [문장]' 형식으로 더 자연스러운 원어민 표현을 제안해 주세요.
    """
    try:
        response = await gemini_model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        return f"코칭 생성 실패: {str(e)}"

async def generate_native_audio(text, file_name):
    """[Azure TTS: 원어민 음성 생성]"""
    if not text or len(text.strip()) < 2: return None
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
    
    audio_path = f"static/audio/{file_name}.mp3"
    audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    
    result = synthesizer.speak_text_async(text).get()
    return f"/static/audio/{file_name}.mp3" if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted else None


# --- [ API 엔드포인트 ] ---

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    recognizer = None
    current_question = "General Conversation"

    try:
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
        stream_format = speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

        pron_config = speechsdk.PronunciationAssessmentConfig(
            reference_text="", 
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            advanced_config={"enableIncompleteAssessment": "true"},
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme
        )

        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        pron_config.apply_to(recognizer)

        while True:
            message = await websocket.receive()
            if "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "stop":
                    current_question = data.get("question", current_question)
                    break
            elif "bytes" in message:
                push_stream.write(message["bytes"])

        push_stream.close()
        result = recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            final_data = process_pronunciation_result(result)
            feedback = await get_ai_coaching(final_data, current_question)
            
            user_tts = await generate_native_audio(result.text, f"u_{result.result_id}")
            
            model_match = re.search(r"Model Answer:\s*(.+)", feedback, re.IGNORECASE)
            model_text = model_match.group(1).strip() if model_match else ""
            model_tts = await generate_native_audio(model_text, f"m_{result.result_id}")

            await websocket.send_text(json.dumps({
                "user_said": result.text,
                "score": final_data['sentence'],
                "words": final_data['words'],
                "feedback": feedback,
                "user_tts_url": user_tts,
                "model_tts_url": model_tts
            }))
    except Exception as e:
        traceback.print_exc()
    finally:
        if recognizer: del recognizer

@app.get("/test")
async def get_test_page():
    return FileResponse("static/test.html")

@app.get("/generate-questions")
async def generate_questions(url: str):
    try:
        transcript = get_transcript_via_whisper(url)
        if not transcript: return {"error": "유튜브 분석 실패"}
        
        prompt = f"""
        다음 영어 스크립트를 분석해 영어 회화 질문 3개를 만들어줘.
        1. 인사말 없이 오직 아래 형식으로만 3줄을 출력해.
        2. 형식: 질문? | (한국어 해석) | [모범 답안: 영어 문장]
        3. 각 질문은 반드시 한 줄에 하나씩 작성해.
        
        스크립트: {transcript[:2500]}
        """
        response = await gemini_model.generate_content_async(prompt)
        return {"questions": response.text}
    except Exception as e:
        return {"error": str(e)}

@app.get("/cumulative-analysis/{user_id}")
async def get_cumulative_report(user_id: int):
    weak_list = get_total_weak_patterns(user_id)
    if not weak_list:
        return {"summary": "아직 충분한 데이터가 쌓이지 않았습니다."}

    prompt = f"""
    사용자의 누적 발음 데이터 분석 결과입니다: {weak_list}
    이 데이터를 바탕으로 사용자의 고질적인 발음 습관을 분석하고, 이를 교정하기 위한 장기적인 훈련 플랜을 한국어로 작성해줘.
    """
    response = await gemini_model.generate_content_async(prompt)
    return {
        "weak_phonemes": weak_list,
        "ai_analysis": response.text
    }   

# --- [ 면접 전용 엔드포인트 ] ---

@app.post("/interview/setup")
async def setup_interview(setup: InterviewSetup):
    # InterviewManager의 generate_initial_question 호출
    first_question = await interview_manager.generate_initial_question(setup.dict())
    return {"status": "success", "question": first_question}

@app.post("/interview/start")
async def start_interview(data: dict):
    mode = data.get("mode") 
    self_intro = data.get("self_intro") 
    user_selection = data.get("user_selection") 
    
    if mode == "ai":
        if self_intro:
            position = data.get("position", "Software Engineer")
            question = await interview_manager.generate_question_from_pdf(self_intro, position)
        elif user_selection:
            question = await interview_manager.generate_initial_question(user_selection)
        else:
            return {"error": "정보가 없습니다."}
    else:
        question = data.get("manual_question", "Please introduce yourself.")
        
    return {"question": question}

@app.post("/interview/start-unified")
async def start_unified_interview(
    position: str = Form(...),
    tech_stack: str = Form(...), # JSON 문자열로 받음
    experience_level: str = Form(...),
    project_summary: str = Form(...),
    interview_mode: str = Form(...),
    file: Optional[UploadFile] = File(None) # PDF는 선택 사항
):
    pdf_text = ""
    if file:
        upload_dir = "temp_uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        try:
            pdf_text = extract_text_from_pdf(file_path)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    # 문자열로 들어온 tech_stack을 리스트로 변환
    import json
    tech_list = json.loads(tech_stack)

    setup_data = {
        "position": position,
        "tech_stack": tech_list,
        "experience_level": experience_level,
        "project_summary": project_summary,
        "interview_mode": interview_mode
    }

    question = await interview_manager.generate_unified_question(setup_data, pdf_text if pdf_text else None)
    return {"status": "success", "question": question}

@app.post("/interview/answer")
async def process_answer(req: AnswerRequest):
    # AnswerRequest 모델을 사용하여 데이터 접근
    follow_up = await interview_manager.generate_follow_up(req.current_question, req.user_answer)
    return {
        "follow_up": follow_up,
        "status": "continue"
    }

@app.post("/interview/upload-pdf")
async def upload_pdf_interview(position: str = Form(...), file: UploadFile = File(...)):
    upload_dir = "temp_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        pdf_text = extract_text_from_pdf(file_path)
        question = await interview_manager.generate_question_from_pdf(pdf_text, position)
        return {
            "status": "success",
            "question": question,
            "extracted_text_preview": pdf_text[:200] + "..."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)