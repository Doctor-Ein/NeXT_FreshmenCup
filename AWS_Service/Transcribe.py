import asyncio
import sounddevice
import numpy as np
from typing import List

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.model import TranscriptEvent
from AWS_Service.config import config

transcribe_streaming = TranscribeStreamingClient(region=config['region'])

class TranscribeService:
    """AWS Transcribe 流式语音转文字服务，仅收集并输出转录文本"""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.is_continuous = False
        self.is_paused = False
        self.mic_stream = None
        self.transcribe_stream = None
        self.transcript_queue = asyncio.Queue(maxsize=10)
        self.continuous_task = None

    async def start(self):
        """启动转录服务"""
        if self.is_continuous:
            print("[Debug]: 服务已启动")
            return

        self.is_continuous = True
        self.mic_stream = MicStream(self.loop, should_accept=lambda: not self.is_paused)

        lc = config['polly']['LanguageCode']
        if lc == 'cmn-CN':
            lc = 'zh-CN'

        self.transcribe_stream = await transcribe_streaming.start_stream_transcription(
            language_code=lc,
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )

        self.continuous_task = asyncio.gather(
            self._write_audio_chunks(),
            self._process_transcripts()
        )

    async def stop(self):
        """停止转录服务"""
        if not self.is_continuous:
            print("[Debug]: 服务未运行")
            return

        self.is_continuous = False
        self.mic_stream.stop()

        try:
            if self.transcribe_stream:
                await self.transcribe_stream.input_stream.end_stream()
        except Exception as e:
            print(f"[Error]: 关闭输入流失败: {e}")

        if self.continuous_task:
            try:
                await self.continuous_task
            except Exception as e:
                print(f"[Error]: 停止时出错: {e}")
            finally:
                self.continuous_task = None
                self.transcribe_stream = None

    def generate_keepalive_noise(duration_ms: int = 20, sample_rate: int = 16000, amplitude: int = 500) -> bytes:
        num_samples = int(sample_rate * duration_ms / 1000)
        # 生成 [-amplitude, +amplitude] 范围内的随机整数
        noise = np.random.randint(-amplitude, amplitude, size=num_samples, dtype=np.int16)
        return noise.tobytes()

    async def _write_audio_chunks(self):
        while self.is_continuous:
            try:
                if self.is_paused:
                    # 发送低音量噪声
                    silence_data = TranscribeService.generate_keepalive_noise(self.mic_stream.block_size)
                    await self.transcribe_stream.input_stream.send_audio_event(audio_chunk=silence_data)
                    await asyncio.sleep(0.05)  # 每 50ms 发送一次低音量数据
                else:
                    async for chunk, _ in self.mic_stream.stream():
                        if not self.is_continuous or self.is_paused:
                            break
                        await self.transcribe_stream.input_stream.send_audio_event(audio_chunk=chunk)
            except Exception as e:
                print(f"[Error]: 发送音频数据时出错: {e}")
                await asyncio.sleep(1)  # 出错时等待 1 秒后重试
        
        # 不要在这里 end_stream，而是在 stop() 中做
        # await self.transcribe_stream.input_stream.end_stream()


    async def _process_transcripts(self):
        """接收并处理转录事件"""
        async for event in self.transcribe_stream.output_stream:
            if isinstance(event, TranscriptEvent):
                if self.is_paused:
                    continue
                results = event.transcript.results
                for result in results:
                    if not result.is_partial:
                        for alt in result.alternatives:
                            if alt.transcript:
                                await self.transcript_queue.put(alt.transcript)

    def pause(self):
        self.is_paused = True
        print("[Debug]: 转录已暂停")

    def resume(self):
        self.is_paused = False
        print("[Debug]: 转录已恢复")

    async def get_transcript(self):
        """获取转录文本（从队列中）"""
        if not self.transcript_queue.empty():
            return await self.transcript_queue.get()
        return None
    
    ## 单次转录模式
    async def one_time_transcription(self, timeout: float = None) -> List[str]:
        """
        单次转录模式：启动后持续转录，直到调用停止时返回所有转录结果
        :param timeout: 可选超时时间(秒)，None表示无超时
        :return: 所有非部分转录结果的列表
        """
        self._reset_transcription_state()
        await self.start()
        
        try:
            if timeout is not None:
                await asyncio.wait_for(self._wait_for_stop(), timeout)
            else:
                await self._wait_for_stop()
        except asyncio.TimeoutError:
            print(f"[Info] 转录已达到超时时间 {timeout}秒")
        finally:
            await self.stop()
            
        return self._get_all_transcripts()

    def _reset_transcription_state(self):
        """重置转录状态"""
        self.one_time_results = []
        self.one_time_stop_event = asyncio.Event()

    async def _wait_for_stop(self):
        """等待停止信号"""
        await self.one_time_stop_event.wait()

    def stop_one_time(self):
        """停止单次转录模式"""
        self.one_time_stop_event.set()

    async def _process_transcripts(self):
        """修改后的处理转录事件方法"""
        async for event in self.transcribe_stream.output_stream:
            if isinstance(event, TranscriptEvent):
                if self.is_paused:
                    continue
                results = event.transcript.results
                for result in results:
                    if not result.is_partial:
                        for alt in result.alternatives:
                            if alt.transcript:
                                # 同时支持队列模式和单次模式
                                await self.transcript_queue.put(alt.transcript)
                                if hasattr(self, 'one_time_results'):
                                    self.one_time_results.append(alt.transcript)

    def _get_all_transcripts(self) -> List[str]:
        """获取所有累积的转录结果"""
        if hasattr(self, 'one_time_results'):
            return self.one_time_results.copy()
        return []

    async def one_time_transcription(self):
        """单次转录流程"""
        self._reset_state()
        await self._start_stream()
        
        try:
            async for chunk in self._mic_stream():
                if not self._running:  # 检查运行状态
                    break
                await self._send_audio(chunk)
                
            return await self._get_final_results()
        finally:
            await self._cleanup()

    async def stop(self):
        """立即停止服务"""
        self._running = False
        if self.mic_stream:
            self.mic_stream.stop()
        if self.transcribe_stream:
            await self.transcribe_stream.input_stream.end_stream()

class MicStream:
    """麦克风输入音频流"""

    def __init__(self, loop, should_accept=lambda: True):
        self.loop = loop
        self.block_size = 512
        self.samplerate = 16000
        self.is_running = True
        self.should_accept = should_accept

    async def stream(self):
        """异步生成音频块"""
        input_queue = asyncio.Queue(maxsize=2)

        import numpy as np

        def callback(indata, frame_count, time_info, status):
            def enqueue():
                try:
                    while not input_queue.empty():
                        input_queue.get_nowait()

                    input_queue.put_nowait((bytes(indata), status))

                except asyncio.QueueFull:
                    print("[Warning]: Queue is full even after clearing.")

            self.loop.call_soon_threadsafe(enqueue)

        with sounddevice.RawInputStream(
            channels=1,
            samplerate=self.samplerate,
            callback=callback,
            blocksize=self.block_size,
            dtype="int16",
            latency="low"
        ):
            while self.is_running:
                try:
                    indata, status = await input_queue.get()
                    yield indata, status
                except Exception as e:
                    print(f"[Error]: 音频流出错: {e}")

    def stop(self):
        self.is_running = False
