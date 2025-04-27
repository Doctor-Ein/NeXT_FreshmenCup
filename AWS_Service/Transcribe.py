import os
import asyncio
import sounddevice

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream

# 支持的语言列表：中文、英语、日语、韩语
# 注意：这里使用的是Transcribe的语言代码，与Polly的语言代码可能不同
voiceLanguageList = ['cmn-CN', 'en-US', 'ja-JP', 'ko-KR']

# 默认配置为中文（voiceLanguageList[0]）
voiceIndex = 0

# 初始化AWS Transcribe服务客户端
# 从环境变量获取AWS区域，默认为us-east-1
aws_region = os.getenv('AWS_REGION', 'us-east-1')
transcribe_streaming = TranscribeStreamingClient(region=aws_region)

class TranscribeHandler(TranscriptResultStreamHandler):
    """处理AWS Transcribe服务返回的转录结果
    
    继承自TranscriptResultStreamHandler，用于处理实时语音转录流
    使用asyncio.Queue在异步环境中传递转录结果
    """
    def __init__(self, transcript_result_stream: TranscriptResultStream):
        super().__init__(transcript_result_stream)
        self.transcript_queue = asyncio.Queue(maxsize=10)  # 限制队列大小
        self.last_partial = ""  # 用于跟踪部分结果
        
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        """处理转录事件的回调方法

        Args:
            transcript_event: AWS Transcribe服务返回的转录事件
        处理流程：
            1. 获取转录结果
            2. 过滤出非部分结果（完整的语音片段）
            3. 将转录文本放入异步队列
        """
        results = transcript_event.transcript.results
        if results:
            for result in results:
                for alt in result.alternatives:
                    if not result.is_partial:
                        # 完整结果，清除部分结果记录
                        self.last_partial = ""
                        await self.transcript_queue.put(alt.transcript)
                    # if result.is_partial:
                    #     # 只有当新的部分结果与上一个不同时才发送
                    #     if alt.transcript != self.last_partial:
                    #         self.last_partial = alt.transcript
                    #         await self.transcript_queue.put(alt.transcript)
                    # else:
                    #     # 完整结果，清除部分结果记录
                    #     self.last_partial = ""
                    #     await self.transcript_queue.put(alt.transcript)

class MicStream:
    """处理麦克风输入流"""
    def __init__(self, is_continuous=False):
        self.is_running = True
        self.is_continuous = is_continuous
        # 优化块大小和采样率
        self.block_size = 512  # 更小的块以减少延迟
        self.samplerate = 44100  # 更高采样率（需与AWS参数匹配）

    async def mic_stream(self):
        """创建麦克风输入流"""
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue(maxsize=2)  # 限制队列大小以减少延迟

        def callback(indata, frame_count, time_info, status): # 回调函数：向输入队列中插入数据和状态的元组
            try:
                loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))
            except asyncio.QueueFull:
                # 如果队列满了，丢弃最旧的数据
                pass

        # 配置音频输入流
        stream = sounddevice.RawInputStream(
            channels=1,
            samplerate=16000,
            callback=callback,
            blocksize=self.block_size,
            dtype="int16",
            latency='low'  # 使用低延迟模式
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
        """将音频数据写入AWS Transcribe流

        Args:
            stream: AWS Transcribe服务的音频流对象
        流程：
            1. 从麦克风流中获取音频数据
            2. 发送到AWS Transcribe服务
            3. 当停止时关闭流
        """
        async for chunk, status in self.mic_stream():
            if self.is_running:
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
        await stream.input_stream.end_stream()

    def stop(self):
        """停止麦克风输入流"""
        self.is_running = False

class TranscribeService:
    """AWS Transcribe服务的主要接口类

    提供：
        1. 语音转录服务的启动和停止
        2. 转录文本的获取
        3. 语言切换功能
        4. 连续音频流的控制
        5. 网页接口的异步生成器
    """
    def __init__(self, language_index=0):
        """初始化转录服务

        Args:
            language_index: 语言索引，默认为0（中文）
        """
        global voiceIndex
        voiceIndex = language_index
        self.mic_stream = None  # 延迟初始化
        self.handler = None
        self.stream = None
        self.continuous_task = None
        self.is_continuous = False

    async def start_transcribe(self):
        """启动转录服务（普通模式）
        
        使用较大的音频块进行处理，适合一般的转录任务
        """
        self.mic_stream = MicStream(is_continuous=False)
        lc = voiceLanguageList[voiceIndex]
        if lc == 'cmn-CN':  # 为中文特判，因为Transcribe和Polly的配置代码不一样😭
            lc = 'zh-CN'

        self.stream = await transcribe_streaming.start_stream_transcription(
            language_code=lc,
            media_sample_rate_hz=44100,
            media_encoding="pcm",
        )
        
        self.handler = TranscribeHandler(self.stream.output_stream)
        asyncio.create_task(self.write_chunks_task())
        asyncio.create_task(self.handler.handle_events())

    async def start_continuous_transcribe(self):
        """启动连续转录服务
        
        使用更小的音频块和更快的处理策略，适合实时语音输入场景
        """
        if self.is_continuous:
            print("连续转录服务已经在运行中")
            return

        self.is_continuous = True
        self.mic_stream = MicStream(is_continuous=True)
        
        lc = voiceLanguageList[voiceIndex]
        if lc == 'cmn-CN':  # 为中文特判，因为Transcribe和Polly的配置代码不一样😭
            lc = 'zh-CN'

        # 启动Transcribe的流，使用优化的配置
        self.stream = await transcribe_streaming.start_stream_transcription(
            language_code=lc,
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )
        
        self.handler = TranscribeHandler(self.stream.output_stream)
        
        # 创建连续处理的任务
        self.continuous_task = asyncio.gather(
            self.write_chunks_task(),
            self.handler.handle_events()  # AWS自带的一个函数
        )

    async def stop_continuous_transcribe(self):
        """停止连续转录服务
        
        优雅地关闭连续转录服务，确保资源正确释放
        """
        if not self.is_continuous:
            print("连续转录服务未在运行")
            return

        self.is_continuous = False
        self.mic_stream.stop()
        
        if self.continuous_task:
            try:
                await self.continuous_task
            except Exception as e:
                print(f"停止连续转录时发生错误: {e}")
            finally:
                self.continuous_task = None
                self.stream = None
                self.handler = None
                # 重新初始化MicStream以备下次使用
                self.mic_stream = MicStream()

    async def write_chunks_task(self):
        """音频块处理任务  

        Args:
            self: 当前实例
        将麦克风输入的音频数据写入AWS Transcribe流
        """
        await self.mic_stream.write_chunks(self.stream)

    async def get_transcript(self):
        """获取转录文本

        Returns:
            str: 转录的文本字符串
            None: 如果处理器未初始化
        Note:
            这是一种异步方法，将等待到新的转录文本到达
        """
        if self.handler:
            return await self.handler.transcript_queue.get()
        return None

    def stop(self):
        """停止转录服务：停止麦克风输入流，从而触发服务的正常关闭"""
        if self.mic_stream:
            self.mic_stream.stop()

    async def change_language(self, language='zh'):
        """更改转录服务的语言
        
        Args:
            language: 语言代码，支持 'zh'(中文), 'en'(英语), 'ja'(日语), 'ko'(韩语)
        """
        language_map = {
            'zh': 0,
            'en': 1,
            'ja': 2,
            'ko': 3
        }
        
        if language not in language_map:
            raise ValueError(f"不支持的语言代码: {language}。支持的语言代码: {list(language_map.keys())}")

        # 如果服务正在运行，需要先停止
        was_running = self.is_continuous
        if was_running:
            await self.stop_continuous_transcribe()
        
        global voiceIndex
        voiceIndex = language_map[language]
        
        # 如果之前在运行，重新启动服务
        if was_running:
            await self.start_continuous_transcribe()

async def main():
    # 1. 初始化服务（默认中文）
    service = TranscribeService()
    
    # 2. 启动连续转录
    print("开始语音转录（按Ctrl+C停止）...")
    await service.start_continuous_transcribe()
    
    # 3. 实时打印结果
    try:
        while True:
            transcript = await service.get_transcript()
            print(transcript,end="")
    except KeyboardInterrupt:
        pass
    finally:
        # 4. 停止服务
        await service.stop_continuous_transcribe()
        print("转录已停止")

if __name__ == "__main__":
    asyncio.run(main())