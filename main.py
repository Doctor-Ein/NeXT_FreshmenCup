import asyncio
from AWS_Service.Transcribe import TranscribeService  # 假设你把上面代码保存在 TranscribeService.py
from AWS_Service.BedrockWrapper import BedrockWrapper

async def run_transcription():
    loop = asyncio.get_event_loop()
    transcriber = TranscribeService(loop)
    bedrock = BedrockWrapper()

    await transcriber.start()
    print("🚀 转录已启动，开始说话吧...")

    history = []
    try:
        for _ in range(30):  # 运行 30 秒，每秒检查一次
            transcript = await transcriber.get_transcript()
            if transcript:
                print(f"[User]: {transcript}")
                transcriber.pause()
                ret = bedrock.invoke_voice(transcript,history)
                history.append({'role':'user','content':{'type':'text','text':transcript}})
                history.append({'role':'assistant','content':{'type':'text','text':ret}})
                transcriber.resume()
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("🛑 手动中断")
    finally:
        await transcriber.stop()
        print("✅ 转录服务已关闭")

if __name__ == "__main__":
    asyncio.run(run_transcription())
