import asyncio
import websockets

async def test_websocket():
    uri = "ws://127.0.0.1:8000/ws/audio"
    
    # 노트북에 있는 아무 .wav 파일이나 경로를 적어줘 (없으면 하나 녹음해봐!)
    file_path = "test_audio.wav"

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ 서버 연결 성공!")
            
            with open(file_path, "rb") as f:
                while True:
                    # 파일을 3200바이트씩 쪼개서 보냄 (실제 스트리밍 느낌)
                    chunk = f.read(3200)
                    if not chunk:
                        break
                    await websocket.send(chunk)
                    await asyncio.sleep(0.1) # 0.1초 간격으로 전송
                    
            print("📤 음성 파일 전송 완료!")
            # 결과가 서버 터미널에 찍히는지 확인해봐!
            await asyncio.sleep(10) 
            
    except Exception as e:
        print(f"❌ 에러: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())

