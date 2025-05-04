import asyncio

from AWS_Service.BedrockWrapper import BedrockWrapper
from AWS_Service.Transcribe import TranscribeService

async def main():
    bedrock_wrapper = BedrockWrapper()  # 假设你已经有一个实现
    service = TranscribeService(bedrock_wrapper, asyncio.get_event_loop())

    print("开始语音转录（按Ctrl+C停止）...")
    await service.start_continuous_transcribe()

    try:
        while True:
            transcript = await service.get_transcript()
            print(f"\n[user]: {transcript}")
    except KeyboardInterrupt:
        pass
    finally:
        await service.stop_continuous_transcribe()
        print("转录已停止")

if __name__ == "__main__":
    asyncio.run(main())