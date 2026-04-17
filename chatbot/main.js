import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers';

// Configuration
const MODEL_NAME = 'HuggingFaceTB/SmolLM2-135M-Instruct';
const HISTORY_EXPIRY_MS = 60 * 60 * 1000; // 1 hour
const STACK_NAME = 'smolchat_history';
const STACK_TIME_NAME = 'smolchat_last_active';

// DOM Elements
const loadingOverlay = document.getElementById('loading-overlay');
const loadingStatus = document.getElementById('loading-status');
const loadingDetail = document.getElementById('loading-detail');
const progressBar = document.getElementById('progress-bar');
const messagesArea = document.getElementById('messages-area');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const deviceLabel = document.getElementById('device-label');
const statusDot = document.getElementById('status-dot');

let generator = null;
let chatHistory = [];

/**
 * Initialize the application
 */
async function init() {
    setupEventListeners();
    loadHistory();
    await initModel();
}

/**
 * Set up DOM event listeners
 */
function setupEventListeners() {
    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = userInput.scrollHeight + 'px';
    });

    // Send on Enter (but not Shift+Enter)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    sendBtn.addEventListener('click', handleSend);
}

/**
 * Load history from localStorage with 1-hour expiry logic
 */
function loadHistory() {
    const lastActive = localStorage.getItem(STACK_TIME_NAME);
    const now = Date.now();

    if (lastActive && (now - parseInt(lastActive)) > HISTORY_EXPIRY_MS) {
        console.log('Chat history expired (>1 hour). Clearing...');
        localStorage.removeItem(STACK_NAME);
        localStorage.removeItem(STACK_TIME_NAME);
        chatHistory = [];
    } else {
        const savedHistory = localStorage.getItem(STACK_NAME);
        if (savedHistory) {
            chatHistory = JSON.parse(savedHistory);
            renderHistory();
        }
    }
}

/**
 * Save history to localStorage
 */
function saveHistory() {
    localStorage.setItem(STACK_NAME, JSON.stringify(chatHistory));
    localStorage.setItem(STACK_TIME_NAME, Date.now().toString());
}

/**
 * Render saved history to the UI
 */
function renderHistory() {
    // Clear initial greeting if history exists
    if (chatHistory.length > 0) {
        messagesArea.innerHTML = '';
        chatHistory.forEach(msg => {
            appendMessage(msg.role, msg.content, false);
        });
    }
}

/**
 * Initialize Transformers.js model
 */
async function initModel() {
    try {
        loadingStatus.innerText = 'Loading Model...';
        
        // Progress callback for model download
        const progress_callback = (data) => {
            if (data.status === 'progress') {
                const progress = data.progress.toFixed(2);
                progressBar.style.width = `${progress}%`;
                loadingDetail.innerText = `Downloading ${data.file}: ${progress}%`;
            } else if (data.status === 'done') {
                loadingDetail.innerText = `Finished loading ${data.file}`;
            } else if (data.status === 'ready') {
                loadingDetail.innerText = 'Model is ready!';
            }
        };

        // Try WebGPU first
        try {
            generator = await pipeline('text-generation', MODEL_NAME, {
                device: 'webgpu',
                progress_callback,
            });
            deviceLabel.innerText = 'WebGPU Accelerated';
            statusDot.style.background = '#10b981'; // Green
            statusDot.style.boxShadow = '0 0 8px #10b981';
        } catch (gpuError) {
            console.warn('WebGPU failed or not supported, falling back to CPU:', gpuError);
            generator = await pipeline('text-generation', MODEL_NAME, {
                device: 'cpu',
                progress_callback,
            });
            deviceLabel.innerText = 'CPU (Fallback Mode)';
            statusDot.style.background = '#f59e0b'; // Amber
            statusDot.style.boxShadow = '0 0 8px #f59e0b';
        }

        // Hide loading overlay
        loadingOverlay.classList.add('hidden');
        sendBtn.disabled = false;
        userInput.focus();

    } catch (error) {
        console.error('Error initializing model:', error);
        loadingStatus.innerText = 'Failed to load model';
        loadingDetail.innerText = 'Please check console for errors and ensure you have a modern browser.';
    }
}

/**
 * Handle sending a message
 */
async function handleSend() {
    const text = userInput.value.trim();
    if (!text || !generator) return;

    // Disable UI
    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Add user message to history and UI
    chatHistory.push({ role: 'user', content: text });
    appendMessage('user', text);
    saveHistory();

    // Create a placeholder for the bot message
    const botMessageDiv = appendMessage('assistant', '');
    let fullResponse = '';

    try {
        // Run inference
        const output = await generator(chatHistory, {
            max_new_tokens: 512,
            temperature: 0.7,
            do_sample: true,
            // Stream the output if supported in the specific version, 
            // but for simplicity here we handle the result
            // Transformers.js v3 supports passing an Iterable for streaming if using specific patterns,
            // but for this vanilla app, we'll wait for the result or use a streaming callback if available.
        });

        // Transformers.js text-generation with chat templates returns an array
        fullResponse = output[0].generated_text.at(-1).content;
        
        // Update UI
        botMessageDiv.innerText = fullResponse;
        chatHistory.push({ role: 'assistant', content: fullResponse });
        saveHistory();

    } catch (error) {
        console.error('Inference error:', error);
        botMessageDiv.innerText = 'Error: Failed to generate response.';
    } finally {
        sendBtn.disabled = false;
        messagesArea.scrollTo(0, messagesArea.scrollHeight);
    }
}

/**
 * Append a message to the chat area
 */
function appendMessage(role, content, animate = true) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message');
    msgDiv.classList.add(role === 'user' ? 'message-user' : 'message-bot');
    if (!animate) msgDiv.style.animation = 'none';
    
    msgDiv.innerText = content;
    messagesArea.appendChild(msgDiv);
    
    // Auto-scroll to bottom
    messagesArea.scrollTo(0, messagesArea.scrollHeight);
    
    return msgDiv;
}

// Start the app
init();
