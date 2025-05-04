import asyncio
from AWS_Service.BedrockWrapper import BedrockWrapper
from AWS_Service.Transcribe import TranscribeService
from AWS_Service.config import config

async def main():
    # 实例化你的 BedrockWrapper（假设它已经正确实现）
    bedrock_wrapper = BedrockWrapper()

    # 实例化 TranscribeService
    transcribe_service = TranscribeService(bedrock_wrapper, asyncio.get_event_loop())

    # 启动连续转录服务
    await transcribe_service.start_continuous_transcribe()

    try:
        while True:
            # 异步获取转录文本
            transcript = await transcribe_service.get_transcript()
            if transcript:
                print(transcript, end='')
            await asyncio.sleep(0.1)  # 为了避免频繁的队列访问，稍作等待
    except asyncio.CancelledError:
        print("[Info]: 转录服务已取消")
    except Exception as e:
        print(f"[Error]: 发生错误 - {e}")
    finally:
        # 停止转录服务
        await transcribe_service.stop_continuous_transcribe()

# 运行测试用例
if __name__ == "__main__":
    asyncio.run(main())