document.addEventListener('DOMContentLoaded', (event) => {
    const askButton = document.querySelector('.ask-button');
    const sendButton = document.querySelector('.send-button');
    const chatInput = document.querySelector('.chat-input');

    // Initially disable send button and chat input
    disableChat();

    askButton.addEventListener('click', () => {
        handleAskAIClick();
    });

    sendButton.addEventListener('click', () => {
        handleAskAIClick();
    });

    chatInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            handleAskAIClick();
        }
    });
});

function handleAskAIClick() {
    const materialNameInput = document.querySelector('.material-name-input');
    const manufacturerInput = document.querySelector('.manufacturer-input');
    const workContentInput = document.querySelector('.work-content-input');

    // Clear the chat window
    const chatWindow = document.querySelector('.chat-messages');
    chatWindow.innerHTML = '';

    showSpinner();

    fetch('/v1/ask-viridium-ai', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ material_name: materialNameInput.value, manufacturer_name: manufacturerInput.value, work_content: workContentInput.value })
        })
        .then(response => {
            if (!response.ok){
                throw new Error("Error in network response");
            }
            return response.json();
        })
        .then(data => {
            console.log(data.result);
            displayMessage('AI', data.result);
            enableChat();
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage('Error', 'An error occurred while getting the response from AI.');
        })
        .finally(() => {
            hideSpinner();
        });
}

function handleRetryClick() {
    const chatInput = document.querySelector('.chat-input');
    const message = chatInput.value.trim();

    if (message !== '') {
        displayMessage('User', message);
        chatInput.value = '';
    }

    showSpinner();

    fetch('/v1/retry-ask-viridium-ai', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message })
        })
        .then(response => {
            if (!response.ok){
                throw new Error("Error in network response");
            }
            return response.json();
        })
        .then(data => {
            console.log(data.response);
            displayMessage('AI', data.result);
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage('Error', 'An error occurred while getting the response from AI.');
        })
        .finally(() => {
            hideSpinner();
            disableChat();
        });
}

function displayMessage(sender, message) {
    const chatWindow = document.querySelector('.chat-messages');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');

    if (typeof message === 'object') {
        messageElement.innerHTML = `<strong>${sender}:</strong><br>${formatMessage(message)}`;
    } else {
        messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    }

    chatWindow.appendChild(messageElement);
    chatWindow.scrollTop = chatWindow.scrollHeight; // Scroll to the bottom
}

function formatMessage(data) {
    let formattedMessage = '<div class="formatted-message">';
    for (const [key, value] of Object.entries(data)) {
        formattedMessage += `<div><strong>${formatKey(key)}:</strong> ${formatValue(value)}</div>`;
    }
    formattedMessage += '</div>';
    return formattedMessage;
}

function formatKey(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, char => char.toUpperCase());
}

function formatValue(value) {
    if (Array.isArray(value)) {
        return `<ul>${value.map(item => `<li>${item}</li>`).join('')}</ul>`;
    }
    return value !== null ? value : 'N/A';
}

function enableChat() {
    const sendButton = document.querySelector('.send-button');
    const chatInput = document.querySelector('.chat-input');

    sendButton.disabled = false;
    chatInput.disabled = false;
}

function disableChat() {
    const sendButton = document.querySelector('.send-button');
    const chatInput = document.querySelector('.chat-input');

    sendButton.disabled = true;
    chatInput.disabled = true;
}

function showSpinner() {
    document.getElementById('spinnerOverlay').style.display = 'flex';
}

function hideSpinner() {
    document.getElementById('spinnerOverlay').style.display = 'none';
}
