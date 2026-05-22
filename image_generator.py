from google import generativeai as genai


class PersonaImageGenerator:

    def __init__(self):
        self.model = genai.GenerativeModel(
    'models/gemini-3.1-flash-lite-preview'
)

    async def generate_persona_prompt(self, interview_history):

        analysis_prompt = f"""
        당신은 전문 면접관이자 이미지 생성 전문가입니다.

        아래의 면접 기록을 분석해서
        가장 잘 어울리는 동물 페르소나를 정해주세요.

        [면접 기록]
        {interview_history}

        반드시 아래 형식만 출력하세요.

        Animal: [동물]
        Reason: [이유]
        Prompt: [영문 프롬프트]
        """

        response = await self.model.generate_content_async(
            analysis_prompt
        )

        text = response.text

        animal = "Unknown"
        reason = "No reason generated."
        prompt = ""

        for line in text.splitlines():

            if line.startswith("Animal:"):
                animal = line.replace("Animal:", "").strip()

            elif line.startswith("Reason:"):
                reason = line.replace("Reason:", "").strip()

            elif line.startswith("Prompt:"):
                prompt = line.replace("Prompt:", "").strip()

        return {
            "animal": animal,
            "reason": reason,
            "prompt": prompt
        }