import json
import datetime

class DB:

    def __init__(self, route = "db.json"):
        self.route = route
        self.data = {"Dialogue": dict(), "Turn_Num": 0, "Turn": dict()}
        try:
            with open(route, "r") as f:
                self.data = json.load(f)
        except FileNotFoundError:
            with open(route, "w") as f:
                json.dump(self.data, f, indent=4)

    def update(self):
        with open(self.route, "w") as f:
            json.dump(self.data, f, indent=4)

    def new_dialogue(self, title = ""):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.data["Dialogue"][timestamp] = {"Title": title, "Turn_ID_List": []}
        self.update()
        return timestamp

    def new_turn(self, timestamp, userInput, reply, filePath):
        self.data["Turn_Num"] += 1
        id = self.data["Turn_Num"]
        self.data["Turn"][id] = {"Domain": timestamp, "userInput": userInput, "reply": reply, "filePath": filePath}
        self.data["Dialogue"][timestamp]["Turn_ID_List"].append(id)
        self.update()
        return id

    def get_dialogue_timestamps(self):
        return self.data["Dialogue"].keys()
    
    def get_dialogue_titles(self):
        return [self.data["Dialogue"][timestamp]["Title"] for timestamp in self.data["Dialogue"].keys()]
    
    def get_dialogue(self, arg):
        for timestamp in self.data["Dialogue"].keys():
            if timestamp == arg or self.data["Dialogue"][timestamp]["Title"] == arg:
                return [self.data["Turn"][id] for id in self.data["Dialogue"][timestamp]["Turn_ID_List"]]
            
    def get_turn(self, id):
        return self.data["Turn"][id]
    
    def get_dialogue_num(self):
        return len(self.data["Dialogue"])
    
    def get_turn_num(self):
        return len(self.data["Turn"])
    
    def traverse(self):
        return self.data["Turn"]
    
    def delete_dialogue(self, arg):
        for timestamp in self.data["Dialogue"].keys():
            if timestamp == arg or self.data["Dialogue"][timestamp]["Title"] == arg:
                for id in self.data["Dialogue"][timestamp]["Turn_ID_List"]:
                    self.delete_turn(id)
                self.data["Dialogue"].pop(timestamp)
        self.update()

    def delete_turn(self, id):
        self.data["Turn"].pop(id)
        self.update()

if __name__ == "__main__":
    database = DB()