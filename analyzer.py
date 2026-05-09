# analyzer.py
import sqlite3 # 혹은 사용하는 DB 라이브러리

def get_total_weak_patterns(user_id):
    """
    DB에서 해당 사용자의 역대 모든 음소 데이터를 가져와 통계를 냄.
    """
    # 1. DB 연결 (예시: SQLite)
    conn = sqlite3.connect('speaking_test.db')
    cursor = conn.cursor()

    # 2. 음소별 평균 점수와 실패 횟수(60점 미만) 집계 쿼리
    query = """
        SELECT 
            phoneme, 
            AVG(score) as avg_score, 
            COUNT(*) as total_count,
            SUM(CASE WHEN score < 60 THEN 1 ELSE 0 END) as fail_count
        FROM pronunciation_results
        WHERE user_id = ?
        GROUP BY phoneme
        HAVING total_count >= 5 -- 데이터 신뢰도를 위해 최소 5회 이상 등장한 음소만
    """
    cursor.execute(query, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    # 3. 분석 결과 리스트 생성
    analysis_list = []
    for row in rows:
        phoneme, avg_score, total_count, fail_count = row
        fail_rate = (fail_count / total_count) * 100
        
        analysis_list.append({
            "phoneme": phoneme,
            "avg_score": round(avg_score, 1),
            "fail_rate": round(fail_rate, 1),
            "total_count": total_count
        })

    # 4. 가중치 기반 정렬 (실패율이 높을수록 상위 노출)
    # 가중치 공식 예시: $Score = FailRate \times 0.7 + (100 - AvgScore) \times 0.3$
    weak_patterns = sorted(analysis_list, key=lambda x: x['fail_rate'], reverse=True)[:5]
    
    return weak_patterns