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

        # PDF 내용이 있을 경우와 없을 경우를 구분하여 프롬프트 구성
        pdf_context = f"\n[추가 정보: 자기소개서 내용]\n{pdf_text}" if pdf_text else "\n(자기소개서 없음: 제공된 기본 정보에만 집중하세요.)"

        prompt = f"""
        당신은 전문 기술 면접관입니다. 아래 제공된 지원자의 상세 정보와 자기소개서(있는 경우)를 바탕으로 첫 번째 영어 질문을 하세요.

        [지원자 기본 정보]
        - 직무: {position}
        - 주요 기술: {tech_stack}
        - 경력 수준: {exp_level}
        - 프로젝트 요약: {project_summary}
        - 면접 스타일: {interview_mode}
        {pdf_context}

        [질문 가이드라인]
        1. 반드시 영어(English)로만 질문하세요.
        2. 자기소개서 내용이 있다면 그 안의 구체적인 경험을 우선적으로 파고드세요. 
        3. 자기소개서가 없다면 프로젝트 요약과 기술 스택을 바탕으로 {exp_level}에 맞는 난이도의 질문을 하세요.
        4. {interview_mode} 스타일의 톤을 유지하며, 서론 없이 바로 질문을 시작하세요.
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

    async def generate_follow_up(self, prev_question, user_answer):
        prompt = f"당신은 면접관입니다. 질문({prev_question})에 대한 답변({user_answer})을 듣고 영어로 꼬리 질문을 하세요."
        response = await self.model.generate_content_async(prompt)
        return response.text.strip()