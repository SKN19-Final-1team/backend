import asyncio
import websockets

async def test_audio_stream():
    uri = "ws://127.0.0.1:8000/api/v1/ws"
        
    async with websockets.connect(uri) as websocket:
        print("테스트 웹소켓 연결 성공")

        dummy_audio_data = open('test.webm', 'rb').read()
        
        try:
            # 데이터 전송
            await websocket.send(dummy_audio_data)
            
            response = await websocket.recv()
            print(f"서버 응답 : {response}")
            
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(test_audio_stream())