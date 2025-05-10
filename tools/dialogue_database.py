import json
from typing import Dict, List, Optional
import uuid
import datetime

class DialogueDB:
    def __init__(self, file_path: str = "dialogue_db.json"):
        self.file_path = file_path
        self._initialize_data()
        self._load_db()
    
    def _initialize_data(self):
        """确保数据结构始终有效"""
        self.data = {
            "version": "1.0",
            "dialogues": {},
            "turns": {},
            "indexes": {
                "dialogue_timestamps": [],
                "dialogue_titles": {},
                "dialogue_turns": {}
            }
        }
    
    def _load_db(self):
        """加载数据并验证结构"""
        try:
            with open(self.file_path, "r") as f:
                loaded_data = json.load(f)
                
                # 验证并修复数据结构
                if not isinstance(loaded_data, dict):
                    raise ValueError("Invalid data format")
                
                # 确保所有必需的键都存在
                for key in ["dialogues", "turns", "indexes"]:
                    if key not in loaded_data:
                        loaded_data[key] = self._initialize_data()[key]
                
                # 确保indexes内的结构正确
                indexes = loaded_data.get("indexes", {})
                for subkey in ["dialogue_timestamps", "dialogue_titles", "dialogue_turns"]:
                    if subkey not in indexes:
                        indexes[subkey] = self._initialize_data()["indexes"][subkey]
                
                self.data = loaded_data
                
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            # 如果文件不存在或数据损坏，初始化新数据
            self._initialize_data()
            self._save_db()
    
    def _save_db(self):
        """保存数据到文件"""
        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=2)
    
    # 其他方法保持不变...

    def _generate_id(self) -> str:
        """Generate a standardized UUID-based ID with timestamp prefix"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = uuid.uuid4().hex[:8]  # First 8 chars of UUID
        return f"{timestamp}_{unique_id}"

    def create_dialogue(self, title: str = "") -> str:
        """创建新对话（确保初始化所有必要结构）"""
        dialogue_id = self._generate_id()
        
        # 初始化数据结构
        if "dialogues" not in self.data:
            self.data["dialogues"] = {}
        if "turns" not in self.data:
            self.data["turns"] = {}
        if "indexes" not in self.data:
            self.data["indexes"] = {
                "dialogue_timestamps": [],
                "dialogue_titles": {},
                "dialogue_turns": {}
            }
        
        # 添加对话数据
        now = datetime.datetime.now().isoformat()
        self.data["dialogues"][dialogue_id] = {
            "title": title,
            "created_at": now,
            "updated_at": now
        }
        
        # 更新索引
        self.data["indexes"]["dialogue_timestamps"].append(dialogue_id)
        self.data["indexes"]["dialogue_turns"][dialogue_id] = []  # 初始化轮次列表
        
        if title:
            self.data["indexes"]["dialogue_titles"][title] = dialogue_id
        
        self._save_db()
        return dialogue_id


    def add_turn(self, dialogue_id: str, speaker: str, content: str, images: List[str] = None) -> str:
        """添加对话轮次（确保索引存在）"""
        # 确保对话存在
        if dialogue_id not in self.data["dialogues"]:
            raise ValueError(f"对话 {dialogue_id} 不存在")
        
        # 确保索引结构存在
        if "indexes" not in self.data:
            self.data["indexes"] = {
                "dialogue_timestamps": [],
                "dialogue_titles": {},
                "dialogue_turns": {}
            }
        
        # 确保该对话的轮次列表存在
        if dialogue_id not in self.data["indexes"]["dialogue_turns"]:
            self.data["indexes"]["dialogue_turns"][dialogue_id] = []
        
        # 生成轮次ID
        turn_id = self._generate_id()
        
        # 添加轮次数据
        self.data["turns"][turn_id] = {
            "dialogue_id": dialogue_id,
            "speaker": speaker,
            "content": content,
            "images": images or [],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # 更新索引
        self.data["indexes"]["dialogue_turns"][dialogue_id].append(turn_id)
        
        # 更新对话的修改时间
        self.data["dialogues"][dialogue_id]["updated_at"] = datetime.datetime.now().isoformat()
        
        self._save_db()
        return turn_id

    def get_dialogue_metadata(self) -> List[Dict[str, str]]:
        """Get all dialogues' metadata in reverse chronological order"""
        return [
            {
                "id": dialogue_id,
                "title": self.data["dialogues"][dialogue_id]["title"],
                "created_at": self.data["dialogues"][dialogue_id]["created_at"],
                "updated_at": self.data["dialogues"][dialogue_id]["updated_at"]
            }
            for dialogue_id in reversed(self.data["indexes"]["dialogue_timestamps"])
        ]

    def get_turns_in_dialogue(self, dialogue_id: str) -> List[Dict]:
        """Get all turns in a dialogue in chronological order"""
        if dialogue_id not in self.data["indexes"]["dialogue_turns"]:
            return []
        
        return [
            self.data["turns"][turn_id]
            for turn_id in self.data["indexes"]["dialogue_turns"][dialogue_id]
        ]

    def search_dialogues_by_title(self, title_query: str) -> List[Dict[str, str]]:
        """Search dialogues by title (case-insensitive partial match)"""
        results = []
        for title, dialogue_id in self.data["indexes"]["dialogue_titles"].items():
            if title_query.lower() in title.lower():
                results.append({
                    "id": dialogue_id,
                    "title": title,
                    "created_at": self.data["dialogues"][dialogue_id]["created_at"]
                })
        return results

    def delete_dialogue(self, dialogue_id: str):
        """Delete a dialogue and all its turns"""
        if dialogue_id not in self.data["dialogues"]:
            return
        
        # Remove all turns
        for turn_id in self.data["indexes"]["dialogue_turns"][dialogue_id]:
            self.data["turns"].pop(turn_id, None)
        
        # Remove from indexes
        title = self.data["dialogues"][dialogue_id]["title"]
        if title in self.data["indexes"]["dialogue_titles"]:
            self.data["indexes"]["dialogue_titles"].pop(title)
        
        self.data["indexes"]["dialogue_timestamps"].remove(dialogue_id)
        self.data["indexes"]["dialogue_turns"].pop(dialogue_id)
        
        # Remove dialogue
        self.data["dialogues"].pop(dialogue_id)
        self._save_db()

    def delete_turn(self, turn_id: str):
        """Delete a specific turn"""
        if turn_id not in self.data["turns"]:
            return
        
        dialogue_id = self.data["turns"][turn_id]["dialogue_id"]
        self.data["indexes"]["dialogue_turns"][dialogue_id].remove(turn_id)
        self.data["turns"].pop(turn_id)
        self._save_db()

    def get_all_dialogues_sorted(self) -> List[Dict]:
        """获取所有对话(按创建时间排序，从早到晚)"""
        return [
            {
                "id": dialogue_id,
                "title": self.db.data["dialogues"][dialogue_id]["title"],
                "created_at": self.db.data["dialogues"][dialogue_id]["created_at"]
            }
            for dialogue_id in self.db.data["indexes"]["dialogue_timestamps"]
        ]

    def update_dialogue_title(self, dialogue_id: str, new_title: str) -> bool:
        """Update the title of a dialogue and maintain all indexes.
        
        Args:
            dialogue_id: The ID of the dialogue to update
            new_title: The new title to set
            
        Returns:
            bool: True if the update was successful, False if the dialogue wasn't found
        """
        # Check if dialogue exists
        if dialogue_id not in self.data["dialogues"]:
            return False
        
        # Get the old title for index updating
        old_title = self.data["dialogues"][dialogue_id]["title"]
        
        # Update the dialogue record
        self.data["dialogues"][dialogue_id]["title"] = new_title
        self.data["dialogues"][dialogue_id]["updated_at"] = datetime.datetime.now().isoformat()
        
        # Update the title index if needed
        if old_title in self.data["indexes"]["dialogue_titles"]:
            self.data["indexes"]["dialogue_titles"].pop(old_title)
        
        if new_title:  # Only index non-empty titles
            self.data["indexes"]["dialogue_titles"][new_title] = dialogue_id
        
        self._save_db()
        return True

class DialogueSession:
    def __init__(self, db: DialogueDB, dialogue_id: str = None):
        self.db = db
        self.current_dialogue = dialogue_id
    
    def set_dialogue(self, dialogue_id: str):
        """设置当前对话"""
        if dialogue_id in self.db.data["dialogues"]:
            self.current_dialogue = dialogue_id
            return True
        return False
    
    def create_new_dialogue(self, title: str = "") -> str:
        """创建新对话并设为当前对话"""
        self.current_dialogue = self.db.create_dialogue(title)
        return self.current_dialogue
    
    def get_current_dialogue_id(self) -> str:
        """获取当前对话ID"""
        return self.current_dialogue
    
    def get_current_dialogue_turns(self) -> List[Dict]:
        """获取当前对话的所有轮次(按时间顺序)"""
        if not self.current_dialogue:
            return []
        return self.db.get_turns_in_dialogue(self.current_dialogue)
    
    def add_turn_to_current(self, speaker: str, content: str, images: List[str] = None) -> str:
        """向当前对话添加轮次"""
        if not self.current_dialogue:
            raise ValueError("没有选中任何对话")
        return self.db.add_turn(self.current_dialogue, speaker, content, images or [])



class DialogueManager:
    def __init__(self, db_path: str = "dialogue_db.json"):
        self.db = DialogueDB(db_path)  # 初始化数据库连接
        self.current_dialogue_id = None  # 当前选中的对话ID
    
    # (1) 对话选择功能
    def create_dialogue(self, title: str = "") -> str:
        """创建新对话并设为当前对话"""
        self.current_dialogue_id = self.db.create_dialogue(title)
        return self.current_dialogue_id
    
    def select_dialogue(self, dialogue_id: str) -> bool:
        """选择现有对话"""
        if dialogue_id in self.db.data["dialogues"]:
            self.current_dialogue_id = dialogue_id
            return True
        return False
    
    # (2) 获取对话轮次
    def get_current_turns(self) -> List[Dict]:
        """获取当前对话的所有轮次(时间顺序)"""
        if not self.current_dialogue_id:
            return []
        return self.db.get_turns_in_dialogue(self.current_dialogue_id)
    
    # (3) 添加轮次
    def add_turn(self, speaker: str, content: str, images: Optional[List[str]] = None) -> str:
        """向当前对话添加轮次"""
        if not self.current_dialogue_id:
            raise ValueError("请先创建或选择对话")
        return self.db.add_turn(self.current_dialogue_id, speaker, content, images or [])
    
    # (4) 获取所有对话 - 修正后的版本
    def get_all_dialogues(self) -> List[Dict]:
        """获取所有对话信息(按创建时间排序)"""
        dialogues = []
        for dialogue_id in self.db.data["indexes"]["dialogue_timestamps"]:
            if dialogue_id in self.db.data["dialogues"]:
                meta = self.db.data["dialogues"][dialogue_id]
                dialogues.append({
                    "id": dialogue_id,
                    "title": meta["title"],
                    "created_at": meta["created_at"],
                    "updated_at": meta["updated_at"]
                })
        return dialogues
    
    # 实用功能
    def search_dialogues(self, keyword: str) -> List[Dict]:
        """搜索包含关键字的对话"""
        return [
            dialogue
            for dialogue in self.get_all_dialogues()
            if keyword.lower() in dialogue["title"].lower()
        ]
    
    def delete_current_dialogue(self):
        """删除当前对话"""
        if self.current_dialogue_id:
            self.db.delete_dialogue(self.current_dialogue_id)
            self.current_dialogue_id = None

    def update_title(self, dialogue_id,new_title):
        self.db.update_dialogue_title(dialogue_id, new_title)