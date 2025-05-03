import flask as fl
import werkzeug.utils

import AWS_Service.BedrockWrapper as BW

import os

LLM_local_service = BW.BedrockWrapper()

LLM_web_service = fl.Flask(__name__)
LLM_web_service.secret_key = "BMS : Unswayed, strive to be a more professional version of yourself."
LLM_web_service.config["UPLOAD_FOLDER"] = "uploads"
LLM_web_service.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(LLM_web_service.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    return "." in filename and (filename.rsplit(".", 1)[-1].lower() in ALLOWED_EXTENSIONS)

@LLM_web_service.route("/", methods=["GET", "POST"])
def index():

    if fl.request.method == "POST":

        userInput = fl.request.form.get("userInput")

        fl.flash(userInput, "userInput")

        file = fl.request.files["file"]
        files = []

        if file and allowed_file(file.filename):
            filename = werkzeug.utils.secure_filename(file.filename)
            file_path = os.path.join(LLM_web_service.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            files.append(file_path)
        
        reply = LLM_local_service.invoke_bedrock(userInput, [], files)

        fl.flash(reply, "reply")

    return fl.render_template("index.html")

if __name__ == "__main__":
    LLM_web_service.run(debug=True)