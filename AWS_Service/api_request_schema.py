api_request_list = {
    'amazon.titan-text-express-v1': {
        "modelId": "amazon.titan-text-express-v1",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "inputText": "",
            "textGenerationConfig": {
                "maxTokenCount": 4096,
                "stopSequences": [],
                "temperature": 0,
                "topP": 1
            }
        }
    },
    'amazon.titan-text-lite-v1': {
        "modelId": "amazon.titan-text-lite-v1",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "inputText": "",
            "textGenerationConfig": {
                "maxTokenCount": 4096,
                "stopSequences": [],
                "temperature": 0,
                "topP": 1
            }
        }
    },
    'anthropic.claude-3-5-sonnet-20240620-v1:0':{
        "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "system":"回答均输出markdown格式的文本。对于数学公式，行内使用$...$；独立公式使用$$...$$，并且在其前需要主动换行。RAG模式下，参考资料以json字符串格式给出，注意辨别无关主题，通过元数据给出正确的来源引用",
            "messages": "",
            "max_tokens": 1024,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 0.9,
            "stop_sequences": [
                "\n\nHuman:"
            ],
            "anthropic_version": "bedrock-2023-05-31"
        }
    },
    'anthropic.claude-3-sonnet-20240229-v1:0': {
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "system":"",
            "messages": "",
            "max_tokens": 1024,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 0.9,
            "stop_sequences": [
                "\n\nHuman:"
            ],
            "anthropic_version": "bedrock-2023-05-31"
        }
    },    
    'anthropic.claude-v2:1': {
        "modelId": "anthropic.claude-v2:1",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "prompt": "",
            "max_tokens_to_sample": 300,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": [
                "\n\nHuman:"
            ],
            "anthropic_version": "bedrock-2023-05-31"
        }
    },
    'anthropic.claude-v2': {
        "modelId": "anthropic.claude-v2",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "prompt": "",
            "max_tokens_to_sample": 300,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": [
                "\n\nHuman:"
            ],
            "anthropic_version": "bedrock-2023-05-31"
        }
    },
    'meta.llama3-70b-instruct-v1': {
        "modelId": "meta.llama3-70b-instruct-v1:0",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "prompt": "",
            "max_gen_len": 512,
            "temperature": 0.1,
            "top_p": 0.9
        }
    },    
    'meta.llama3-8b-instruct-v1': {
        "modelId": "meta.llama3-8b-instruct-v1:0",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "prompt": "",
            "max_gen_len": 512,
            "temperature": 0.1,
            "top_p": 0.9
        }
    },        
    'meta.llama2-13b-chat-v1': {
        "modelId": "meta.llama2-13b-chat-v1",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "prompt": "",
            "max_gen_len": 512,
            "temperature": 0.2,
            "top_p": 0.9
        }
    },
    'meta.llama2-70b-chat-v1': {
        "modelId": "meta.llama2-70b-chat-v1",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "prompt": "",
            "max_gen_len": 512,
            "temperature": 0.2,
            "top_p": 0.9
        }
    },
    'mistral.mistral-large-2402-v1:0': {
        "modelId": "mistral.mistral-large-2402-v1:0",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "prompt": "",
            "max_tokens": 1024,
            "temperature": 0.5,
            "top_p": 0.9
        }
    },    
    'cohere.command-text-v14': {
        "modelId": "cohere.command-text-v14",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "prompt": "",
            "max_tokens": 1024,
            "temperature": 0.8,
        }
    },
    'cohere.command-light-text-v14': {
        "modelId": "cohere.command-light-text-v14",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {
            "prompt": "",
            "max_tokens": 1024,
            "temperature": 0.8,
        }
    },
}


def get_model_ids():
    return list(api_request_list.keys())
