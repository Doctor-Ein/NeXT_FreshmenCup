let uploadedImageBase64 = "";
let messageCount = 0;

document.getElementById("configIcon").addEventListener("click", function () {
    const popup = document.getElementById("configPopup");
    popup.style.display = (popup.style.display === "block") ? "none" : "block";
});

function closePopup() {
    document.getElementById("configPopup").style.display = "none";
}

function sendMessage() {
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();
    if (message || uploadedImageBase64) {
        console.log("发送消息:", message);
        userInput.value = '';
        removeImage();
    }
}

function removeImage() {
    const preview = document.getElementById('imagePreview');
    preview.innerHTML = '';
    uploadedImageBase64 = '';
}