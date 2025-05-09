from flask import Flask, request, jsonify, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:8080"])  # 明确指定允许的来源

from dialogue_database import DialogueManager

manager = DialogueManager('./test_db.json')

# 1. 创建新对话
@app.route('/api/create_dialogue', methods=['POST'])
def create_dialogue():
    data = request.get_json()
    title = data.get('title', '')
    timestamp = manager.create_dialogue(title)
    return jsonify({'id': timestamp}),201 


# 2. 获取所有对话列表
@app.route('/api/dialogue_list', methods=['GET'])
def dialogue_list():
    dialogues = manager.get_all_dialogues()
    dialogues.reverse() # 按照从最近到过往的顺序返回
    return jsonify(dialogues)

# 3. 获取会话内的所有轮次的消息
@app.route('/api/get_messages/<dialogue_id>', methods=['GET'])
def update_messages(dialogue_id):
    if manager.select_dialogue(dialogue_id):
        turns = manager.get_current_turns()
        return jsonify(turns),200
    else:
        abort(500, description="dialogue doesn't exist")

@app.route('/api/settings', methods=['POST'])
def model_schema_settings():
    from AWS_Service.config import api_request
    try:
        # 获取前端发送的 JSON 数据
        data = request.get_json()

        # 验证所需字段是否存在
        required_fields = ['temperature', 'top_k', 'top_p', 'max_tokens', 'prompt']
        if not all(field in data for field in required_fields):
            return jsonify({'error': '缺少必要的配置字段'}), 400

        # 更新模型配置
        api_request['body']['temperature'] = data['temperature']
        api_request['body']['top_k'] = data['top_k']
        api_request['body']['top_p'] = data['top_p']
        api_request['body']['max_tokens'] = data['max_tokens']
        api_request['body']['system'] = data['prompt']  # 将 prompt 设置为 system 提示词

        return jsonify({"message": "配置更新成功"}), 200
    except Exception as e:
        return jsonify({'error': f'配置更新失败: {str(e)}'}), 500

isRAGEnabled = False # aaa随手弄的全局变量哭了
# from RAG_Package.QueryEngine import QueryEngine
# query_engine = QueryEngine()

@app.route('/api/rag_toggle', methods=['POST','OPTIONS'])
def rag_toggle():
    global isRAGEnabled
    data = request.get_json()
    isRAGEnabled = data['rag_enabled']
    return jsonify({'status': 'success'}), 200

from AWS_Service.BedrockWrapper import BedrockWrapper 
from image_zip import compress_base64_image
bedrock = BedrockWrapper()

@app.route('/api/submit', methods=['POST'])
def handleSubmit():
    try:
        data = request.get_json()
    except Exception as e:
        print('Error',e)
    manager.add_turn(speaker='user',content=data['text'], images=data['images']) # 这里有个概念命名未对齐的问题🤔content在数据库中仅为text的含义

    ## 这里执行图像的预处理，有些图像需要压缩
    images = [compress_base64_image(item['data'],item['media_type']) for item in data['images']]
    if None in images:
        images = []

    input_text = data['text']
    # ## 这里执行RAG的处理流程
    # if isRAGEnabled:
    #     request_text = ''
    #     out = query_engine.query(input_text, top_k=5, use_rerank=True, rerank_top_k=3)
    #     for item in out:
    #         request_text += item['main'] + '\n'
    #         with open('debug.txt','w',encoding='utf-8') as f:
    #             print(item['main'],file=f,end='\n====================\n')
    request_text = input_text
    dd = manager.get_current_turns()
    dls = [{'role':item['speaker'],'content':[{'type':'text','text':item['content']}]} for item in dd]
    # dls = [{'role':'assistant','content':[{'type':'text','text':'我在上海交通大学读书'}]}]
    response = bedrock.invoke_model(request_text,dialogue_list=dls,images=images)
    manager.add_turn(speaker='assistant',content=response,images=[])
    return jsonify({'res':response}), 200

import sounddevice as sd
import numpy as np
import io
import wave
import sys
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.model import TranscriptEvent
from AWS_Service.config import config

# 音频录制参数
SAMPLERATE = 16000
CHANNELS = 1
BLOCKSIZE = 1024
dtype = np.int16

# 存储音频数据的队列
audio_data = []

def audio_callback(indata, frames, time, status):
    """音频回调函数，将录制的音频数据添加到列表"""
    if status:
        print(status, file=sys.stderr)
    audio_data.append(indata.copy())

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """开始录音"""
    try:
        with sd.InputStream(callback=audio_callback, channels=CHANNELS, samplerate=SAMPLERATE, blocksize=BLOCKSIZE, dtype=dtype):
            print("开始录音...")
            sd.sleep(10000)  # 录制 10 秒钟
            print("录音结束")
        return jsonify({"message": "录音完成"})
    except Exception as e:
        print(f"录音失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """停止录音"""
    print("停止录音")
    return jsonify({"message": "录音已停止"})

@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    """转录音频数据"""
    try:
        # 将音频数据转换为 WAV 格式
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(np.dtype(dtype).itemsize)
            wf.setframerate(SAMPLERATE)
            wf.writeframes(b''.join(audio_data))
        audio_buffer.seek(0)

        # 初始化 TranscribeStreamingClient
        transcribe_client = TranscribeStreamingClient(region=config['region'])

        # 启动转录流
        stream = transcribe_client.start_stream_transcription(
            language_code='zh-CN',
            media_sample_rate_hz=SAMPLERATE,
            media_encoding='pcm',
        )

        # 发送音频数据
        while True:
            chunk = audio_buffer.read(1024)
            if not chunk:
                break
            stream.input_stream.send_audio_event(audio_chunk=chunk)

        # 结束转录流
        stream.input_stream.end_stream()

        # 获取转录结果
        transcript = ''
        for event in stream.output_stream:
            if isinstance(event, TranscriptEvent):
                for result in event.transcript.results:
                    if not result.is_partial:
                        for alt in result.alternatives:
                            transcript += alt.transcript + '\n'

        return jsonify({"transcript": transcript})
    except Exception as e:
        print(f"转录失败: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)


# import asyncio
# from AWS_Service.Transcribe import TranscribeService  # 假设你把上面代码保存在 TranscribeService.py
# from AWS_Service.BedrockWrapper import BedrockWrapper 

# async def run_transcription():
#     loop = asyncio.get_event_loop()
#     transcriber = TranscribeService(loop)
#     bedrock = BedrockWrapper()

#     await transcriber.start()
#     print("🚀 转录已启动，开始说话吧...")

#     history = []
#     try:
#         for _ in range(30):  # 运行 30 秒，每秒检查一次
#             transcript = await transcriber.get_transcript()
#             if transcript:
#                 print(f"[User]: {transcript}")
#                 await transcriber.start()
#                 ret = bedrock.invoke_voice(transcript,history)
#                 history.append({'role':'user','content':{'type':'text','text':transcript}})
#                 history.append({'role':'assistant','content':{'type':'text','text':ret}})
#                 await transcriber.stop()
#                 transcriber.resume()
#             await asyncio.sleep(1)
#     except KeyboardInterrupt:
#         print("🛑 手动中断")
#     finally:
#         await transcriber.stop()
#         print("✅ 转录服务已关闭")

# def main():
#     bedrock = BedrockWrapper()
#     input_text = input('[Prompt]:')
#     response = bedrock.invoke_model(input_text)
#     print(response)

# main()