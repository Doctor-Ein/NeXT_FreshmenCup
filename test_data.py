from dialogue_database import DialogueManager
import time

def generate_test_data(db_path="test_db.json"):
    """生成测试对话数据（安全版本）"""
    print(f"正在生成测试数据到 {db_path}...")
    
    try:
        # 初始化数据库（强制重建）
        manager = DialogueManager(db_path)
        
        # 清除可能存在的旧数据
        manager.db.data = {
            "dialogues": {},
            "turns": {},
            "indexes": {
                "dialogue_timestamps": [],
                "dialogue_titles": {},
                "dialogue_turns": {}
            }
        }
        manager.db._save_db()
        
        # 测试数据生成...
        topics = ["Python学习", "产品支持", "订单咨询"]
        
        for i, topic in enumerate(topics, 1):
            # 创建对话
            dialogue_id = manager.create_dialogue(f"{topic} #{i}")
            print(f"\n创建对话: {topic} #{i} (ID: {dialogue_id})")
            
            # 添加轮次
            for j in range(1, 4):  # 每个对话3个轮次
                speaker = "user" if j % 2 == 1 else "assistant"
                content = f"{speaker}的第{j}条消息关于{topic}"
                images = ["test.png"] if speaker == "assistant" else []
                
                turn_id = manager.add_turn(
                    speaker=speaker,
                    content=content,
                    images=images
                )
                print(f"  添加轮次 {turn_id[:8]}...")
                time.sleep(1)
        
        time.sleep(1)
        # 验证数据
        assert len(manager.db.data["dialogues"]) == len(topics)
        for d_id in manager.db.data["dialogues"]:
            assert d_id in manager.db.data["indexes"]["dialogue_turns"]
        
        print("\n测试数据生成成功！")
        print(f"- 对话数: {len(manager.db.data['dialogues'])}")
        print(f"- 轮次数: {len(manager.db.data['turns'])}")
        
    except Exception as e:
        print(f"\n生成测试数据时出错: {str(e)}")
        raise

if __name__ == '__main__':
    generate_test_data()