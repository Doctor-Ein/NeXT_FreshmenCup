import asyncio
import sounddevice
import sys
from concurrent.futures import ThreadPoolExecutor

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream
from AWS_Service.BedrockWrapper import BedrockWrapper
from AWS_Service.config import config

transcribe_streaming = TranscribeStreamingClient(region=config['region'])

class TranscribeService:
    """AWS Transcribe服务的主要接口类（仅支持连续转录模式）"""

    def __init__(self, bedrock_wrapper: BedrockWrapper, loop: asyncio.AbstractEventLoop):
        self.bedrock_wrapper = bedrock_wrapper
        self.loop = loop
        self.is_continuous = False
        self.is_paused = False  # 新增：暂停标志
        self.mic_stream = None
        self.handler = None
        self.stream = None
        self.continuous_task = None

    async def start_continuous_transcribe(self):
        """启动连续转录服务"""
        if self.is_continuous:
            print("[Debug]: 连续转录服务已在运行中")
            return

        self.is_continuous = True
        self.mic_stream = MicStream(loop=self.loop, is_continuous=True)

        lc = config['polly']['LanguageCode']
        if lc == 'cmn-CN': # 为中文特判，因为Transcribe和Polly的代码不一致
            lc = 'zh-CN'

        # 启动 Transcribe 流
        self.stream = await transcribe_streaming.start_stream_transcription(
            language_code=lc,
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )
        
        self.handler = TranscribeHandler(self.stream.output_stream, self.bedrock_wrapper, self.loop)

        # 创建连续处理任务
        self.continuous_task = asyncio.gather(
            self.write_chunks_task(),
            self.handler.handle_events()  # AWS自带的一个函数
        )

    async def stop_continuous_transcribe(self):
        """停止连续转录服务"""
        if not self.is_continuous:
            print("[Debug]: 连续转录服务未在运行")
            return

        self.is_continuous = False
        self.mic_stream.stop()

        if self.continuous_task:
            try:
                await self.continuous_task
            except Exception as e:
                print(f"[Error]: 停止连续转录时发生错误: {e}")
            finally:
                self.continuous_task = None
                self.stream = None
                self.handler = None
                self.mic_stream = MicStream(loop=self.loop)

    async def write_chunks_task(self):
        """音频块处理任务"""
        await self.mic_stream.write_chunks(self.stream)

    def pause(self):
        """暂停转录服务"""
        self.is_paused = True
        if self.handler:
            self.handler.paused = True
        print("[Debug]: 转录服务已暂停")

    def resume(self):
        """恢复转录服务"""
        self.is_paused = False
        if self.handler:
            self.handler.paused = False
        print("[Debug]: 转录服务已恢复")
    
    async def get_transcript(self):
        """从队列中获取当前的转录文本"""
        if self.handler:
            # 直接使用 await 获取队列中的转录文本
            return await self.handler.get_transcript_from_queue()
        return None

class TranscribeHandler(TranscriptResultStreamHandler):
    """处理AWS Transcribe服务返回的转录结果"""

    text = []
    history = []
    sample_count = 0
    max_sample_counter = 4
    paused = False  # 确保初始化paused属性

    def __init__(self, transcript_result_stream, bedrock_wrapper: BedrockWrapper, loop: asyncio.AbstractEventLoop):
        super().__init__(transcript_result_stream)
        self.transcript_queue = asyncio.Queue(maxsize=10)  # 限制队列大小
        self.bedrock = bedrock_wrapper
        self.loop = loop

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        """处理转录事件的回调方法"""
        if self.paused:
            return  # 暂停状态时跳过处理

        results = transcript_event.transcript.results
        if results:
            for result in results:
                TranscribeHandler.sample_count = 0
                if not result.is_partial:  # 只处理完整的转录结果
                    for alt in result.alternatives:
                        await self.transcript_queue.put(alt.transcript)  # 放入队列
                        
                        # 每隔一定时间或达到样本计数上限时提交文本
                        TranscribeHandler.text.append(alt.transcript)
        else:
            TranscribeHandler.sample_count += 1
            if TranscribeHandler.sample_count >= TranscribeHandler.max_sample_counter:
                input_text = ' '.join(TranscribeHandler.text)

                if len(input_text)!=0:
                    self.loop.run_in_executor(
                        ThreadPoolExecutor(),
                        self.bedrock.invoke_voice,
                        input_text,
                        TranscribeHandler.history
                    )

                # 清空缓存
                TranscribeHandler.text.clear()
                TranscribeHandler.sample_count = 0

    async def get_transcript_from_queue(self):
        """从队列中获取转录文本"""
        if not self.transcript_queue.empty():
            return await self.transcript_queue.get()
        return None  # 如果队列为空，则返回 None


class MicStream:
    """处理麦克风输入流"""

    def __init__(self, loop ,is_continuous=False):
        self.is_running = True
        self.is_continuous = is_continuous
        self.block_size = 512  # 更小的块以减少延迟
        self.samplerate = 16000  # 采样率与 AWS 匹配
        self.loop = loop

    async def mic_stream(self):
        """创建麦克风输入流"""  
        input_queue = asyncio.Queue(maxsize=2)

        def callback(indata, frame_count, time_info, status):
            try:
                self.loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))
            except asyncio.QueueFull:
                pass

        # 配置音频输入流
        stream = sounddevice.RawInputStream(
            channels=1,
            samplerate=16000,
            callback=callback,
            blocksize=self.block_size,
            dtype="int16",
            latency='low'
        )

        with stream:
            while self.is_running:
                try:
                    indata, status = await input_queue.get()
                    yield indata, status
                except Exception as e:
                    print(f"音频流处理错误: {e}")
                    continue

    async def write_chunks(self, stream):
        """将音频数据写入 AWS Transcribe 流"""
        async for chunk, status in self.mic_stream():
            if self.is_running:
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
        await stream.input_stream.end_stream()

    def stop(self):
        """停止麦克风输入流"""
        self.is_running = False

def printer(text: str, level: str) -> None:
    """
    打印日志信息
    功能描述：根据日志级别打印信息，错误信息重定向到 stderr
    :param text: 要打印的文本
    :param level: 日志级别（info或debug）
    """
    if level == 'error':
        print(text, file=sys.stderr)
    elif config['log_level'] == 'info' and level == 'info':
        print(text)
    elif config['log_level'] == 'debug' and level in ['info', 'debug']:
        print(text)
