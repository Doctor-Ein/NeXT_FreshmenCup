import queue
import asyncio
import sounddevice as sd
from concurrent.futures import InvalidStateError 
from AWS_Service.config import config
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent


class MicrophoneStream:
    """实时将麦克风数据放入队列"""
    def __init__(self, rate=16000, chunk_size=1024):
        self.rate = rate
        self.chunk_size = chunk_size
        self._buff = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"录音状态警告：{status}")
        # 将原始 PCM 字节放入队列
        self._buff.put(bytes(indata))

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.rate,
            blocksize=self.chunk_size,
            channels=1,
            dtype='int16',
            callback=self._callback
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
        # 向队列发送终止信号
        self._buff.put(None)

    def generator(self):
        """生成器：不断读取队列中的音频块，直至 None"""
        while True:
            chunk = self._buff.get()
            if chunk is None:
                return
            yield chunk

class TranscribeService:
    """
    基于 Amazon Transcribe Streaming SDK 的单次会话转录服务。
    """
    def __init__(self, region: str = 'us-east-1', language_code: str = 'en-US'):
        self.client = TranscribeStreamingClient(region=region)
        self.language_code = language_code
        self.audio_stream = MicrophoneStream()
        self._transcript_chunks = []

    async def _send_audio(self, stream):
        """并行任务：将麦克风数据发送到 Transcribe 输入流"""
        for chunk in self.audio_stream.generator():
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
        # 结束流
        await stream.input_stream.end_stream()

    async def _receive_transcript(self, stream):
        handler = TranscriptResultStreamHandler(stream.output_stream)

        # 自定义回调：逐条处理完整（非部分）结果
        async def _custom_handler(event: TranscriptEvent):
            try:
                for result in event.transcript.results:
                    if not result.is_partial:
                        self._transcript_chunks.append(result.alternatives[0].transcript)
            except InvalidStateError:
                # Safe to ignore if the SDK tried to set a result on a cancelled Future
                pass

        handler.handle_transcript_event = _custom_handler

        try:
            await handler.handle_events()
        except InvalidStateError:
            # 忽略 SDK 内部回调的状态冲突错误
            pass
        except asyncio.CancelledError:
            # 忽略流结束时被取消的任务
            pass


    async def start_transcription(self):
        """
        启动流式转录会话并麦克风采集。
        返回一个 Task，用于后续停止时等待。
        """
        # 开启麦克风录制
        self.audio_stream.start()
        # 建立流式转录会话
        stream = await self.client.start_stream_transcription(
            language_code=self.language_code,
            media_sample_rate_hz=self.audio_stream.rate,
            media_encoding='pcm'
        )
        # 并行发送与接收
        self._send_task = asyncio.create_task(self._send_audio(stream))
        self._receive_task = asyncio.create_task(self._receive_transcript(stream))

    async def stop_transcription(self) -> str:
        # 1. 停止麦克风录制，触发发送任务结束
        self.audio_stream.stop()

        # 2. 等待发送任务结束，但忽略 CancelledError
        try:
            await self._send_task
        except asyncio.CancelledError:
            pass  # 这里屏蔽取消异常

        # 3. 等待接收任务结束，并同样屏蔽取消异常
        try:
            await self._receive_task
        except asyncio.CancelledError:
            pass

        # 4. 返回累积的转录文本
        return ' '.join(self._transcript_chunks)

async def main():
    svc = TranscribeService(region=config['region'], language_code='zh-CN')
    await svc.start_transcription()
    input("请说话，按回车键停止实时转录…")
    text = await svc.stop_transcription()
    print("===== 转录结果 =====")
    print(text)

if __name__ == '__main__':
    asyncio.run(main())
