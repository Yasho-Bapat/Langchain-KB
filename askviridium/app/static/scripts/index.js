document.addEventListener('DOMContentLoaded', (event) => {
    const askButton = document.querySelector('.ask-button');
    const sendButton = document.querySelector('.send-button');
    const chatInput = document.querySelector('.chat-input');

    askButton.addEventListener('click', () => {
        handleAskAIClick();
    });

    sendButton.addEventListener('click', () => {
        handleClick();
    });

    chatInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            handleClick();
        }
    });
});

function handleAskAIClick() {
    const materialNameInput = document.querySelector('.material-name-input');
    const manufacturerInput = document.querySelector('.manufacturer-input');
    const workContentInput = document.querySelector('.work-content-input');

    fetch('/v1/ask-viridium-ai', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ material_name: materialNameInput.value, manufacturer_name: manufacturerInput.value, work_content: workContentInput.value })
        })
        .then(response => {
            if (!response.ok){
                throw new Error("Error in network repsonse");
            }
            return response.json()
        })
        .then(data => {
            console.log(data.result)
            displayMessage('AI', data.result);
            enableChat();
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage('Error', 'An error occurred while getting the response from AI.');
        });
}

function handleClick() {
    const chatInput = document.querySelector('.chat-input');
    const message = chatInput.value.trim();

    if (message !== '') {
        displayMessage('User', message);
        chatInput.value = '';
    }

    fetch('/v1/ask-viridium-ai', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message })
        })
        .then(response => {
            response.json()
            disableChat();
        })
        .then(data => {
            console.log(data.response)
            displayMessage('AI', data.result);
            disableChat();
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage('Error', 'An error occurred while getting the response from AI.');
        });
}

function displayMessage(sender, message) {
    const chatWindow = document.querySelector('.chat-messages');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatWindow.appendChild(messageElement);
}
