from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许跨域访问

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