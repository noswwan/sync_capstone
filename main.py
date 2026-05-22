import os
import json
import asyncio
import re
import shutil
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from typing import Optional
from image_generator import PersonaImageGenerator
from openai import AsyncOpenAI

# 외부 라이브러리
import azure.cognitiveservices.speech as speechsdk
import google.generativeai as genai
import whisper

# 내부 모듈
from utils import get_transcript_via_whisper, extract_text_from_pdf
from analyzer import get_total_weak_patterns
from interview_manager import InterviewManager
from image_generator import PersonaImageGenerator

# 1. 환경 변수 로드 및 AI 설정
load_dotenv()
AZURE_KEY = os.getenv('AZURE_SPEECH_KEY')
AZURE_REGION = os.getenv('AZURE_SPEECH_REGION')
GEMINI_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('models/gemini-3.1-flash-lite-preview')
interview_manager = InterviewManager(gemini_model)

app = FastAPI(title="Sync Capstone API")
interview_session_history = []

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
    
@app.get("/interview-test")
async def get_interview_test():
    return FileResponse("static/interview_test.html")

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

# 2. [수정] 면접 시작 엔드포인트 수정 (새 면접이 시작되면 기존 기록 비우기)
@app.post("/interview/start-unified")
async def start_unified_interview(
    position: str = Form(...),
    tech_stack: str = Form(...), # JSON 문자열로 받음
    experience_level: str = Form(...),
    project_summary: str = Form(...),
    interview_mode: str = Form(...),
    file: Optional[UploadFile] = File(None) # PDF는 선택 사항
):
    # 1. 새로운 면접이 시작되므로 전역 기록을 완전히 초기화해줌
    global interview_session_history
    interview_session_history = []
    
    # 2. [복구] PDF 파일이 있을 경우 파일 저장 후 텍스트 추출하는 로직
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

    # 3. 문자열로 들어온 tech_stack을 리스트로 변환
    import json
    tech_list = json.loads(tech_stack)

    setup_data = {
        "position": position,
        "tech_stack": tech_list,
        "experience_level": experience_level,
        "project_summary": project_summary,
        "interview_mode": interview_mode
    }

    # 4. 첫 번째 영어 질문 생성
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

stt_model = whisper.load_model("base") # 모델 로드

    
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

image_gen = PersonaImageGenerator()

@app.post("/generate-image")
async def generate_image(chat_history: list):
    # 1. 대화 기록을 텍스트로 합치기
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])
    
    # 2. AI에게 동물 분석 및 프롬프트 요청
    result = await image_gen.generate_persona_prompt(history_text)
    
    # 3. (실제 구현 시) 여기서 DALL-E API 등을 호출해 이미지 URL을 받음
    # 지금은 테스트를 위해 생성된 프롬프트와 분석 내용을 반환할게
    return {"result": result}

# main.py 내 answer_voice 함수 내부 저장 로직 확인
@app.post("/interview/answer-voice")
async def answer_voice(
    audio: UploadFile = File(...), 
    current_question: str = Form(...)
):
    global interview_session_history
    try:
        # 1. 파일 저장
        file_path = f"static/audio/{audio.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        # 2. 로컬 파일로 STT 실행 (내 진짜 목소리 텍스트 변환)
        transcription = get_transcript_via_whisper(file_path)
        
        if not transcription or len(transcription.strip()) < 2:
            transcription = "I'm sorry, I couldn't hear you clearly."

        # 3. 꼬리 질문 생성
        follow_up = await interview_manager.generate_follow_up(current_question, transcription)
        
        # 🌟 4. [신규] 내 답변을 원어민 음성용 올바른 영어 문장으로 교정 (유저가 원하는 알맞은 발음 텍스트)
        pronunciation_guide = "I'm sorry, I couldn't process the correct sentence."
        if transcription != "I'm sorry, I couldn't hear you clearly.":
            correct_prompt = f"""
            You are a native English editor. 
            Correct and rewrite the candidate's broken or casual answer into a natural, grammatically perfect native sentence.
            Maintain the original meaning but make it professional for an interview.
            
            [STRICT RULE]
            Output ONLY the corrected English sentence. No explanations, no corrections notes.
            
            Candidate's Answer: "{transcription}"
            """
            correct_response = await gemini_model.generate_content_async(correct_prompt)
            pronunciation_guide = correct_response.text.strip()

        # 🌟 5. 데이터 누적 (이제 가이드 칸에 진짜 원어민 교정 문장이 들어감)
        interview_session_history.append({
            "question": current_question,
            "transcription": transcription,         # 내가 말한 텍스트 그대로 (여러 줄 지원)
            "pronunciation_guide": pronunciation_guide, # 내 대답에 대한 알맞은 원어민 표현 문장!
            "user_audio_url": f"/static/audio/{audio.filename}" # 내 진짜 목소리 녹음 파일
        })
        
        return {
            "status": "success",
            "transcription": transcription,
            "follow_up": follow_up
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/interview/finalize")
async def finalize_interview():
    global interview_session_history
    try:
        # 1. [안전장치] 진행된 대화 기록이 없을 때 예외 처리
        if not interview_session_history:
            return {
                "status": "error",
                "animal_reason": "진행된 면접 답변 데이터가 존재하지 않아 리포트를 생성할 수 없습니다."
            }
        
        # 2. 실시간 면접 질문과 유저 답변 텍스트를 하나로 융합
        full_transcript = ""
        for idx, item in enumerate(interview_session_history):
            full_transcript += f"질문 {idx+1}: {item['question']}\n답변 {idx+1}: {item['transcription']}\n\n"
        
        # 3. Gemini에게 성향 분석 및 맞춤형 동물 페르소나 추출 요청 (프롬프트 보완)
        analysis_prompt = f"""
        You are an expert technical interview recruiter and behavior analyst. 
        Analyze the following technical interview transcript to determine a unique 'Animal Persona' for the candidate based on their communication style, technical depth, and confidence.
        
        [Interview Transcript]
        {full_transcript}
        
        [Instructions]
        1. Select an animal that perfectly matches the candidate's technical responding style.
        2. Combine it with a professional adjective in Korean (e.g., '치밀한 고양이', '날카로운 독수리').
        3. Provide a detailed reason in Korean explaining why this animal fits them.
        4. Provide an English prompt fragment suitable for generating a profile image of this animal as an anthropomorphic technical developer (e.g., 'a methodical cat developer', 'a sharp eagle engineer').
        
        [Strict Rule]
        Return the output ONLY as a valid JSON string. Do NOT include markdown code block formatting (like ```json).
        
        [Output Format]
        {{
            "animal_name": "형용사 + 동물 이름",
            "animal_reason": "상세한 분석 및 선정 이유 (한국어)",
            "animal_generation_prompt": "영어 이미지 생성용 핵심 구문"
        }}
        """
        
        # 4. AI 모델 호출 및 데이터 파싱
        response = await gemini_model.generate_content_async(analysis_prompt)
        result_text = response.text.strip()
        
        # JSON 파싱 가드 로직
        if "```" in result_text:
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        analysis_data = json.loads(result_text.strip())
        
        animal_name = analysis_data.get("animal_name", "신중한 올빼미")
        animal_reason = analysis_data.get("animal_reason", "기술적 질문에 신중하게 답변하셨습니다.")
        # AI가 분석한 이미지 생성용 핵심 구문 가져오기 (예: 'a practical beaver developer')
        animal_generation_prompt = analysis_data.get("animal_generation_prompt", "an owl developer").strip()
        
        # 🌟🌟🌟 5. [핵심 업데이트] AI(DALL-E 3)가 직접 이미지를 생성하는 로직 구현 🌟🌟🌟
        # URL 가져오는 게 아니라 코딩을 통해 AI가 직접 동물을 의인화해서 그려냄
        
        print(f"🎨 AI 이미지 생성 시작 (DALL-E 3 모델 호출): {animal_generation_prompt}...")
        
        try:
            # DALL-E 3 비동기 생성 요청
            image_generation_response = await openai_client.images.generate(
                model="dall-e-3", # 최고 사양의 생성 모델 사용
                prompt=f"""
                A detailed square profile picture of {animal_generation_prompt} wearing a cozy hoodie and holding a coffee mug, focused on a MacBook screen displaying code. 
                Anthropomorphic style with high quality digital art. 
                Modern tech startup developer office background with subtle blue LED lighting. 
                Cinematic lighting, cute yet professional expression, 8k resolution, rendered in Octane.
                """,
                n=1,
                size="1024x1024", # 고해상도 정사각형 이미지
                quality="hd" # 고품질 생성
            )
            
            # 생성된 고유 이미지의 임시 웹 주소 추출
            animal_image_url = image_generation_response.data[0].url
            print("✅ AI 이미지 실시간 생성 성공!")
            
        except Exception as img_err:
            print(f"⚠️ AI 이미지 생성 실패 (기본 이미지 대체): {img_err}")
            animal_image_url = "[https://via.placeholder.com/400x400?text=AI+Image+Pending](https://via.placeholder.com/400x400?text=AI+Image+Pending)" # 실패 시 대기 이미지

        # 6. 기술 스택 언급 비중 계산 로직 연동
        tech_counts = {"FastAPI": 0, "Python": 0, "WebSocket": 0}
        total_text = full_transcript.lower()
        for tech in tech_counts.keys():
            tech_counts[tech] = total_text.count(tech.lower())
        
        total_mentions = sum(tech_counts.values())
        if total_mentions > 0:
            tech_stack_percent = {k: round((v / total_mentions) * 100) for k, v in tech_counts.items()}
        else:
            tech_stack_percent = {"FastAPI": 50, "Python": 30, "WebSocket": 20}
            
        return {
            "status": "success",
            "animal_name": animal_name,
            "animal_image_url": animal_image_url, # 🌟 AI가 방금 따끈따끈하게 그린 고유 주소 전송
            "animal_reason": animal_reason,
            "content_improvement": f"전반적으로 {animal_name} 성향에 맞는 몰입도 높은 답변이었습니다. 가끔 실제 프로젝트 벤치마크 지표를 명확히 제시한다면 더 완벽한 포지셔닝이 가능합니다.",
            "tech_stack_percent": tech_stack_percent,
            "interview_history": interview_session_history
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "animal_reason": f"종합 리포트 데이터 분석 및 AI 이미지 직접 생성 중 에러가 발생했습니다: {str(e)}"
        }