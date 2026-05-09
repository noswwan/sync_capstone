from sqlalchemy import create_password, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 아까 도커에서 설정한 정보들이야
SQLALCHEMY_DATABASE_URL = "postgresql://geunhan:password123@localhost:5432/sync_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()