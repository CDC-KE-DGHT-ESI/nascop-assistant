document.addEventListener('DOMContentLoaded', function() {
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');
    const typingIndicator = document.getElementById('typing-indicator');
    const exampleButtons = document.querySelectorAll('.example-button');
    const menuToggle = document.getElementById('menu-toggle');
    const sidebar = document.getElementById('sidebar');
    const conversationList = document.getElementById('conversation-list');
    const newChatBtn = document.getElementById('new-chat-btn');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const initialTimeElement = document.getElementById('initial-time');
    
    // Set distinctive page title with emoji for tab identification
    document.title = "NASCOP Assistant";
    
    // Text-to-Speech functionality
    let speechSynthesis = window.speechSynthesis;
    let isSpeechEnabled = false;
    let currentUtterance = null;
    
    // Function to initialize text-to-speech
    function initTextToSpeech() {
        // Check if browser supports speech synthesis
        if (!('speechSynthesis' in window)) {
            console.error('Your browser does not support speech synthesis');
            return false;
        }
        return true;
    }
    
    // Function to toggle speech functionality
    function toggleSpeech() {
        isSpeechEnabled = !isSpeechEnabled;
        
        // Update button appearance
        const speechToggleBtn = document.getElementById('speech-toggle-btn');
        if (isSpeechEnabled) {
            speechToggleBtn.classList.add('active');
            speechToggleBtn.setAttribute('title', 'Turn off voice');
        } else {
            speechToggleBtn.classList.remove('active');
            speechToggleBtn.setAttribute('title', 'Turn on voice');
            // Stop any ongoing speech
            if (speechSynthesis.speaking) {
                speechSynthesis.cancel();
            }
        }
        
        // Save preference to localStorage
        localStorage.setItem('nascop_speech_enabled', isSpeechEnabled);
    }
    
    // Function to speak text
    function speakText(text) {
        if (!isSpeechEnabled || !initTextToSpeech()) return;
        
        // First, cancel any ongoing speech
        if (speechSynthesis.speaking) {
            speechSynthesis.cancel();
        }
        
        // Clean the text more thoroughly for speech synthesis
        // First, handle specific markdown patterns before removing HTML tags
        let cleanText = text
            // Replace raw markdown patterns before HTML cleaning
            .replace(/\*\*/g, '') // Remove all ** for bold
            .replace(/\n\s*\*\s+/g, ', ') // Convert asterisk bullet points to natural pauses
            .replace(/\n\s*\+\s+/g, ', ') // Convert plus bullet points to natural pauses
            .replace(/\n\s*\d+\.\s+/g, ', number ') // Convert numbered lists to natural speech
            // Now remove HTML tags that may have been applied by formatMarkdown
            .replace(/<[^>]*>/g, ' ')
            // Normalize whitespace
            .replace(/\s+/g, ' ')
            .trim();
        
        // Create speech utterance
        const utterance = new SpeechSynthesisUtterance(cleanText);
        
        // Set properties
        utterance.lang = 'en-US';
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        
        // Store current utterance for potential cancellation
        currentUtterance = utterance;
        
        // Speak
        speechSynthesis.speak(utterance);
        
        // Add event listener for when speech ends
        utterance.onend = function() {
            currentUtterance = null;
        };
    }
    
    // Function to stop speech
    function stopSpeech() {
        if (speechSynthesis.speaking) {
            speechSynthesis.cancel();
        }
    }
    
    // Set initial message time
    initialTimeElement.textContent = formatTime(new Date());
    
    // API endpoint and access key
    const apiEndpoint = 'https://agent-8d4577d7737ec53f9b24-joien.ondigitalocean.app/api/v1/chat/completions';
    const accessKey = '0AYgDA6KofWNIqf-8NycxbevxUn1WLNk';
    
    // Track active conversation
    let activeConversationId = generateId();
    let conversations = {};
    
    // Initialize conversations from localStorage if available
    loadConversations();
    
    // Update examples visibility based on initial conversation
    updateExamplesVisibility();
    
    // Animate the typing indicator ellipsis
    function animateEllipsis() {
        const typingIndicator = document.getElementById('typing-indicator');
        let dotCount = 0;
        let baseText = "Thinking";
        
        // Clear any existing interval
        if (window.ellipsisInterval) {
            clearInterval(window.ellipsisInterval);
        }
        
        // Create new interval for animation
        window.ellipsisInterval = setInterval(() => {
            let dots = ".".repeat(dotCount + 1);
            typingIndicator.textContent = baseText + dots;
            dotCount = (dotCount + 1) % 3; // Cycle through 1, 2, 3 dots
        }, 400); // Change dots every 400ms
    }
    
    // Format time
    function formatTime(date) {
        return new Intl.DateTimeFormat('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        }).format(date);
    }
    
    // Generate a unique ID
    function generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }
    
    // Load conversations from localStorage
    function loadConversations() {
        const savedConversations = localStorage.getItem('nascop_conversations');
        if (savedConversations) {
            conversations = JSON.parse(savedConversations);
            updateConversationList();
        }
        
        // If no conversations exist, create a new one
        if (Object.keys(conversations).length === 0) {
            createNewConversation();
        } else {
            // Set the most recent conversation as active
            const conversationIds = Object.keys(conversations);
            if (conversationIds.length > 0) {
                const mostRecentId = conversationIds.sort((a, b) => {
                    return conversations[b].lastUpdated - conversations[a].lastUpdated;
                })[0];
                
                // For backwards compatibility - ensure hideExamples property exists
                conversationIds.forEach(id => {
                    if (conversations[id].hideExamples === undefined) {
                        conversations[id].hideExamples = false;
                    }
                });
                
                setActiveConversation(mostRecentId);
            }
        }
    }
    
    // Save conversations to localStorage
    function saveConversations() {
        localStorage.setItem('nascop_conversations', JSON.stringify(conversations));
    }
    
    // Create a new conversation
    function createNewConversation() {
        const newId = generateId();
        const now = Date.now();
        
        conversations[newId] = {
            id: newId,
            title: 'New Conversation',
            messages: [
                {
                    role: "assistant",
                    content: "Hello! I can help answer questions about NASCOP guidelines and HIV treatment protocols. What would you like to know?",
                    time: now
                }
            ],
            created: now,
            lastUpdated: now,
            hideExamples: false // Flag to keep examples hidden throughout this conversation
        };
        
        saveConversations();
        updateConversationList();
        setActiveConversation(newId);
        
        // Clear chat messages and add initial message
        chatMessages.innerHTML = '';
        addMessage(
            "Hello! I'm the NASCOP Virtual Assistant. I can help answer questions about NASCOP guidelines and HIV treatment protocols. What would you like to know?",
            false,
            now
        );
    }
    
    // Update the conversation list in sidebar
    function updateConversationList() {
        conversationList.innerHTML = '';
        
        // Sort conversations by last updated time (newest first)
        const sortedConversations = Object.values(conversations).sort((a, b) => 
            b.lastUpdated - a.lastUpdated
        );
        
        sortedConversations.forEach(conv => {
            const item = document.createElement('div');
            item.classList.add('conversation-item');
            if (conv.id === activeConversationId) {
                item.classList.add('active');
            }
            
            // Get title from first user message or use default
            let title = conv.title;
            if (title === 'New Conversation') {
                const firstUserMessage = conv.messages.find(m => m.role === 'user');
                if (firstUserMessage) {
                    title = firstUserMessage.content.substring(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '');
                    // Update conversation title
                    conversations[conv.id].title = title;
                    saveConversations();
                }
            }
            
            const titleSpan = document.createElement('span');
            titleSpan.textContent = title;
            item.appendChild(titleSpan);
            
            // Add delete button for individual conversation
            const deleteBtn = document.createElement('button');
            deleteBtn.classList.add('delete-btn');
            deleteBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>';
            deleteBtn.title = "Delete this conversation";
            
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent triggering the parent click
                deleteConversation(conv.id);
            });
            
            item.appendChild(deleteBtn);
            item.dataset.id = conv.id;
            
            item.addEventListener('click', (e) => {
                if (e.target !== deleteBtn && !deleteBtn.contains(e.target)) {
                    setActiveConversation(conv.id);
                }
            });
            
            conversationList.appendChild(item);
        });
    }
    
    // Delete a specific conversation
    function deleteConversation(conversationId) {
        if (confirm("Are you sure you want to delete this conversation?")) {
            delete conversations[conversationId];
            saveConversations();
            
            // If we deleted the active conversation, create a new one
            if (conversationId === activeConversationId) {
                createNewConversation();
            } else {
                updateConversationList();
            }
        }
    }
    
    // Clear all conversation history
    function clearAllConversations() {
        if (confirm("Are you sure you want to clear all conversation history? This action cannot be undone.")) {
            conversations = {};
            saveConversations();
            createNewConversation();
        }
    }
    
    // Set the active conversation and load its messages
    function setActiveConversation(conversationId) {
        if (!conversations[conversationId]) return;
        
        activeConversationId = conversationId;
        updateConversationList();
        
        // Stop any ongoing speech when switching conversations
        stopSpeech();
        
        // Load conversation messages
        chatMessages.innerHTML = '';
        conversations[conversationId].messages.forEach(msg => {
            // Don't use typing animation for loading previous messages
            addMessage(msg.content, msg.role === 'user', msg.time);
        });
        
        // Update examples visibility based on conversation status
        updateExamplesVisibility();
        
        // On mobile, close the sidebar after selection
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('open');
        }
    }
    
    // Update visibility of example buttons based on conversation state
    function updateExamplesVisibility() {
        const examplesContainer = document.querySelector('.examples');
        
        // If this conversation was started as "New Chat", keep examples hidden
        if (conversations[activeConversationId].hideExamples) {
            examplesContainer.style.display = 'none';
        } else {
            examplesContainer.style.display = 'flex';
        }
    }
    
    // Add message to the active conversation
    function addMessageToConversation(message, isUser, timestamp = Date.now()) {
        if (!conversations[activeConversationId]) return;
        
        conversations[activeConversationId].messages.push({
            role: isUser ? "user" : "assistant",
            content: message,
            time: timestamp
        });
        
        conversations[activeConversationId].lastUpdated = timestamp;
        saveConversations();
        updateConversationList();
        
        // Hide examples after user sends their first message
        if (conversations[activeConversationId].messages.filter(msg => msg.role === "user").length === 1) {
            conversations[activeConversationId].hideExamples = true;
            updateExamplesVisibility();
        }
    }
    
    // Add message to the chat UI
    function addMessage(message, isUser, timestamp = Date.now()) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        messageElement.classList.add(isUser ? 'user-message' : 'bot-message');
        
        const contentElement = document.createElement('div');
        contentElement.classList.add('message-content');
        
        if (isUser) {
            contentElement.textContent = message;
        } else {
            // Format markdown-style text
            contentElement.innerHTML = formatMarkdown(message);
            
            // Add play button for bot messages
            const playButton = document.createElement('button');
            playButton.classList.add('play-speech-btn');
            playButton.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></svg>';
            playButton.title = "Read aloud";
            
            playButton.addEventListener('click', () => {
                speakText(message);
            });
            
            messageElement.appendChild(playButton);
            
            // If speech is enabled, automatically speak bot messages
            if (isSpeechEnabled) {
                // Slight delay to ensure DOM is updated
                setTimeout(() => {
                    speakText(message);
                }, 100);
            }
        }
        
        const timeElement = document.createElement('div');
        timeElement.classList.add('message-time');
        timeElement.textContent = formatTime(new Date(timestamp));
        
        messageElement.appendChild(contentElement);
        messageElement.appendChild(timeElement);
        
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return { messageElement, contentElement };
    }
    
    // Simulate typing effect for bot messages
    function simulateTyping(message, timestamp = Date.now()) {
        // Create an empty message bubble first
        const { messageElement, contentElement } = addMessage('', false, timestamp);
        
        // Format the message with markdown before typing animation
        const formattedMessage = formatMarkdown(message);
        
        // Type the message character by character
        let charIndex = 0;
        let currentText = '';
        
        // Set consistently fast typing speed
        const typingSpeed = 10; // Very fast typing speed (10ms per character)
        
        function typeNextChar() {
            if (charIndex < formattedMessage.length) {
                // Handle HTML tags appropriately
                if (formattedMessage[charIndex] === '<') {
                    // Find the end of the tag
                    const endTagIndex = formattedMessage.indexOf('>', charIndex);
                    if (endTagIndex !== -1) {
                        const tag = formattedMessage.substring(charIndex, endTagIndex + 1);
                        currentText += tag;
                        contentElement.innerHTML = currentText;
                        charIndex = endTagIndex + 1;
                    } else {
                        charIndex++;
                    }
                } else {
                    currentText += formattedMessage[charIndex];
                    contentElement.innerHTML = currentText;
                    charIndex++;
                }
                
                // Use consistent fast speed
                setTimeout(typeNextChar, typingSpeed);
                
                // Auto-scroll as typing occurs
                chatMessages.scrollTop = chatMessages.scrollHeight;
            } else {
                // Typing complete, ensure the full formatted message is set
                contentElement.innerHTML = formattedMessage;
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                // Add play button now that typing is complete
                const playButton = document.createElement('button');
                playButton.classList.add('play-speech-btn');
                playButton.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></svg>';
                playButton.title = "Read aloud";
                
                playButton.addEventListener('click', () => {
                    speakText(message);
                });
                
                messageElement.appendChild(playButton);
                
                // If speech is enabled, automatically speak when typing is complete
                if (isSpeechEnabled) {
                    speakText(message);
                }
            }
        }
        
        // Start typing with a small initial delay
        setTimeout(typeNextChar, 300);
    }
    
    // Format markdown-style text
    function formatMarkdown(text) {
        return text
            // Numbered lists
            .replace(/(\d+)\.\s+\*\*([^*]+)\*\*:/g, '<strong>$1. $2:</strong>')
            // Bold text
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            // Bullet points with asterisks
            .replace(/\n\s*\*\s+([^\n]+)/g, '<li>$1</li>')
            // Bullet points with plus signs
            .replace(/\n\s*\+\s+([^\n]+)/g, '<li>$1</li>')
            // Wrap bullet points in ul tags
            .replace(/(<li>.*?<\/li>(\s*<li>.*?<\/li>)*)/gs, '<ul>$1</ul>')
            // Paragraph breaks
            .replace(/\n\n/g, '<br><br>')
            // Single line breaks (excluding list items)
            .replace(/\n(?!<li>)/g, '<br>');
    }
    
    // Send message to the API
    async function sendMessageToAPI(message) {
        typingIndicator.style.display = 'block';
        animateEllipsis(); // Start the animation
        
        try {
            // Prepare API request with conversation history
            const messages = conversations[activeConversationId].messages.map(msg => ({
                role: msg.role,
                content: msg.content
            }));
            
            const response = await fetch(apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${accessKey}`
                },
                body: JSON.stringify({
                    messages: messages,
                    stream: false,
                    include_functions_info: false,
                    include_retrieval_info: false,
                    include_guardrails_info: false
                })
            });
            
            if (!response.ok) {
                throw new Error(`API request failed with status: ${response.status}`);
            }
            
            const data = await response.json();
            typingIndicator.style.display = 'none';
            
            // Stop the animation when we get the response
            if (window.ellipsisInterval) {
                clearInterval(window.ellipsisInterval);
            }
            
            // Process the response
            if (data && data.choices && data.choices.length > 0 && data.choices[0].message) {
                const botResponse = data.choices[0].message.content;
                const timestamp = Date.now();
                simulateTyping(botResponse, timestamp);
                addMessageToConversation(botResponse, false, timestamp);
            } else {
                console.error("Unexpected API response format:", data);
                const errorMsg = "Sorry, I received a response in an unexpected format.";
                const timestamp = Date.now();
                simulateTyping(errorMsg, timestamp);
                addMessageToConversation(errorMsg, false, timestamp);
            }
        } catch (error) {
            console.error('Error:', error);
            typingIndicator.style.display = 'none';
            
            // Stop the animation if there's an error
            if (window.ellipsisInterval) {
                clearInterval(window.ellipsisInterval);
            }
            
            const errorMsg = `Sorry, there was an error: ${error.message}`;
            const timestamp = Date.now();
            addMessage(errorMsg, false, timestamp);
            addMessageToConversation(errorMsg, false, timestamp);
        }
    }
    
    // Handle sending a message
    function handleSendMessage() {
        const message = messageInput.value.trim();
        if (message) {
            const timestamp = Date.now();
            addMessage(message, true, timestamp);
            addMessageToConversation(message, true, timestamp);
            messageInput.value = '';
            sendMessageToAPI(message);
        }
    }
    
    // Event listeners
    sendButton.addEventListener('click', handleSendMessage);
    
    messageInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            handleSendMessage();
        }
    });
    
    // Example buttons
    exampleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const exampleText = this.textContent;
            messageInput.value = exampleText;
            handleSendMessage();
        });
    });
    
    // New chat button
    newChatBtn.addEventListener('click', function() {
        stopSpeech(); // Stop any ongoing speech
        createNewConversation();
    });
    
    // Clear history button
    clearHistoryBtn.addEventListener('click', clearAllConversations);
    
    // Mobile menu toggle
    menuToggle.addEventListener('click', function() {
        sidebar.classList.toggle('open');
    });
    
    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', function(event) {
        if (window.innerWidth <= 768 && 
            !sidebar.contains(event.target) && 
            !menuToggle.contains(event.target) &&
            sidebar.classList.contains('open')) {
            sidebar.classList.remove('open');
        }
    });
    
    // Create and append speech toggle button to the header
    const header = document.querySelector('.header');
    const speechToggleBtn = document.createElement('button');
    speechToggleBtn.id = 'speech-toggle-btn';
    speechToggleBtn.classList.add('speech-toggle-btn');
    speechToggleBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></svg>';
    speechToggleBtn.title = "Turn on voice";
    header.appendChild(speechToggleBtn);
    
    // Initialize speech setting from localStorage
    const savedSpeechSetting = localStorage.getItem('nascop_speech_enabled');
    if (savedSpeechSetting === 'true') {
        isSpeechEnabled = true;
        speechToggleBtn.classList.add('active');
        speechToggleBtn.title = "Turn off voice";
    }
    
    // Add event listener to speech toggle button
    speechToggleBtn.addEventListener('click', toggleSpeech);
    
    // Add event listeners to conversation items to stop speech when switching
    document.addEventListener('click', function(event) {
        if (event.target.closest('.conversation-item')) {
            stopSpeech();
        }
    });
    
    // Initialize text-to-speech
    initTextToSpeech();
});