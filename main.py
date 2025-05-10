from flask import Flask, request, jsonify, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:8080"])  # 明确指定允许的来源

from tools.dialogue_database import DialogueManager

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

@app.route('/api/update_title',methods=['POST'])
def update_title():
    data = request.get_json()
    invoke_text = "为下面的对话总结摘要一个标题，字数限制10个汉字以内：\n" + str([{'speaker':item['speaker'],'content':item['content']} for item in data['content']])
    temp=config['bedrock']['api_request']['body']['max_tokens']
    config['bedrock']['api_request']['body']['max_tokens']=32 # 临时改成极小的最长输出
    ret = bedrock.invoke_model(invoke_text)
    manager.update_title(data['id'], ret)
    config['bedrock']['api_request']['body']['max_tokens']=temp
    return jsonify({'status':'success'}),200

from AWS_Service.Polly import Reader
reader: Reader

@app.route('/api/read',methods=['POST','GET'])
def read_content():
    global reader
    if request.method == 'POST':
        data = request.get_json()
        reader = Reader(data['content'])
        reader.start()
        return jsonify({'status':'success'}),200
    elif request.method == 'GET':
        reader.stop()
        reader.join()
        del reader
        return jsonify({'status':'success'}),200

import threading
import queue
import asyncio  # 新增导入
from AWS_Service.Transcribe import TranscribeService
from AWS_Service.config import config

# 1.1 指令队列与结果存储
command_queue = queue.Queue()
result_queue = queue.Queue()

# 1.2 后台线程包装函数（修复异步运行问题）
def transcribe_worker_wrapper():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(transcribe_worker())

async def transcribe_worker():
    svc = None  # 每次会话独立实例
    while True:
        cmd = command_queue.get()
        if cmd == 'start':
            # 终止之前的服务（如果有）
            if svc is not None:
                await svc.stop_transcription()
            svc = TranscribeService(region=config['region'], language_code='zh-CN')
            await svc.start_transcription()
        elif cmd == 'stop':
            if svc is not None:
                text = await svc.stop_transcription()
                result_queue.put(text)
                svc = None  # 清理实例
        elif cmd == 'exit':
            if svc is not None:
                await svc.stop_transcription()
            break
        command_queue.task_done()

# 1.3 启动后台线程（使用包装函数）
worker_thread = threading.Thread(target=transcribe_worker_wrapper, daemon=True)
worker_thread.start()

@app.route('/api/transcribe', methods=['POST', 'GET'])
def toggle_transcribe():
    # 最小会话隔离（示例，需完善）
    if request.method == 'POST':
        command_queue.put('start')
        return jsonify({'status': 'started', 'tip': 'Call GET to get result'}), 200

    elif request.method == 'GET':
        command_queue.put('stop')
        try:
            # 添加超时避免永久阻塞
            text = result_queue.get(timeout=30)
        except queue.Empty:
            return jsonify({'error': 'Timeout waiting for transcription'}), 504
        return jsonify({'text': text}), 200

isRAGEnabled = False # aaa随手弄的全局变量哭了
# from RAG_Package.QueryEngine import query_engine

@app.route('/api/rag_toggle', methods=['POST','OPTIONS'])
def rag_toggle():
    global isRAGEnabled
    data = request.get_json()
    isRAGEnabled = data['rag_enabled']
    return jsonify({'status': 'success'}), 200

from AWS_Service.BedrockWrapper import BedrockWrapper 
from tools.image_zip import compress_base64_image
bedrock = BedrockWrapper()

@app.route('/api/submit', methods=['POST'])
def handleSubmit():
    data = request.get_json()
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
    response = bedrock.invoke_model(request_text,dialogue_list=dls,images=images)
    manager.add_turn(speaker='assistant',content=response,images=[])
    return jsonify({'res':response}), 200

if __name__ == '__main__':
    app.run(debug=True)