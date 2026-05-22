import google.generativeai as genai

class InterviewManager:
    def __init__(self, model):
        self.model = model

    # [통합] 기본 정보 + PDF(선택)를 조합하여 첫 질문 생성
    async def generate_unified_question(self, setup_data, pdf_text=None):
        position = setup_data.get('position')
        tech_stack = ", ".join(setup_data.get('tech_stack', []))
        exp_level = setup_data.get('experience_level')
        project_summary = setup_data.get('project_summary')
        interview_mode = setup_data.get('interview_mode')

        pdf_context = f"\n[추가 정보: 자기소개서 내용]\n{pdf_text}" if pdf_text else "\n(자기소개서 없음)"

        prompt = f"""
        You are a professional technical interviewer. Based on the information below, ask the FIRST interview question in English.

        [Candidate Info]
        - Job Position: {position}
        - Key Tech: {tech_stack}
        - Exp Level: {exp_level}
        - Project: {project_summary}
        - Style: {interview_mode}
        {pdf_context}

        [STRICT RULES]
        1. Output ONLY the interview question in English.
        2. Do NOT include any Korean, introductions, or meta-commentary.
        3. Start directly with the question.
        4. Maintain a {interview_mode} tone.
        """
        response = await self.model.generate_content_async(prompt)
        return response.text.strip()

    # [수정] 클래스 내부 정렬 및 예외 처리 가이드라인 강화
    async def generate_follow_up(self, prev_question, user_answer):
        # 음성 인식 결과가 부실하거나 실패했을 때의 대응
        if not user_answer or len(user_answer.strip()) < 5 or "음성 인식 실패" in user_answer:
            prompt = f"The candidate's answer was unclear or missing. Politely ask them to repeat or clarify their response to: '{prev_question}' in English."
        else:
            prompt = f"""
            You are a professional technical interviewer. 
            Based on the candidate's answer: "{user_answer}" to your question: "{prev_question}", ask a concise follow-up question in English.
            
            [STRICT RULES]
            1. Output ONLY the interview question in English.
            2. Do NOT include any meta-commentary, explanations, or 'interview intent'.
            3. Do NOT break character as an interviewer.
            4. Keep the question technical and focused on the candidate's previous response.
            """
        
        response = await self.model.generate_content_async(prompt)
        return response.text.strip()

    # 기존 메서드 유지 (무결성 보장)
    async def generate_initial_question(self, user_selection):
        position = user_selection.get('position', 'Software Engineer')
        tech_list = ", ".join(user_selection.get('tech_stack', []))
        project = user_selection.get('project', 'Experience')
        prompt = f"당신은 기술 면접관입니다. 지원자의 직무({position}), 기술({tech_list}), 프로젝트({project})를 바탕으로 첫 번째 영어 면접 질문을 하나만 하세요."
        response = await self.model.generate_content_async(prompt)
        return response.text.strip()

    async def generate_question_from_pdf(self, pdf_text, position):
        prompt = f"당신은 전문 기술 면접관입니다. 자기소개서({pdf_text})를 읽고 직무({position}) 관련 영어 질문을 하세요."
        response = await self.model.generate_content_async(prompt)
        return response.text.strip()