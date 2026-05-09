from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import SQLModel, Field, create_engine, Relationship

# --- [1] 사용자 계정 테이블 ---
class User(SQLModel, table=True):
    __tablename__ = "users"
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str
    nickname: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# --- [2] 전체 발음 학습 기록 (문장 단위) ---
class SpeakingLog(SQLModel, table=True):
    __tablename__ = "speaking_logs"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id")
    
    reference_text: str       # AI가 제시한 원문 (예: "I like soccer")
    recognized_text: Optional[str] = None  # 사용자가 실제로 말한 내용
    
    accuracy_score: float     # 정확도 (얼마나 정확한 단어를 썼나)
    pronunciation_score: float # 발음 점수 (원어민과 얼마나 비슷한가)
    fluency_score: float      # 유창성 (얼마나 매끄럽게 말했나)
    
    coaching_message: Optional[str] = None # Gemini가 준 전체 피드백 저장
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# --- [3] 단어별 세부 기록 (단어 단위) ---
class WordLog(SQLModel, table=True):
    __tablename__ = "word_logs"
    id: Optional[int] = Field(default=None, primary_key=True)
    speaking_log_id: int = Field(foreign_key="speaking_logs.id")
    
    word: str                 # 분석된 단어 (예: "soccer")
    accuracy_score: float     # 해당 단어의 정확도 점수
    error_type: Optional[str] = None  # 오류 유형 (None, Omission, Insertion 등)
    
    # 💡 [알고리즘용 핵심 데이터] 음소(Phoneme)별 점수를 JSON 형태로 저장
    # 예: [{"ph": "s", "score": 90}, {"ph": "aa", "score": 45}]
    phoneme_data: Optional[str] = None 
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# --- DB 연결 설정 ---
DATABASE_URL = "postgresql://geunhan:password123@localhost:5432/sync_db"
engine = create_engine(DATABASE_URL)