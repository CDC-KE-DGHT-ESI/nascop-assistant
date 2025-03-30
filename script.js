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
    const initialTimeElement = document.getElementById('initial-time');
    
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
            
            item.textContent = title;
            item.dataset.id = conv.id;
            
            item.addEventListener('click', () => {
                setActiveConversation(conv.id);
            });
            
            conversationList.appendChild(item);
        });
    }
    
    // Set the active conversation and load its messages
    function setActiveConversation(conversationId) {
        if (!conversations[conversationId]) return;
        
        activeConversationId = conversationId;
        updateConversationList();
        
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
            // Bullet points
            .replace(/\n\s*\*\s+([^\n]+)/g, '\n<li>$1</li>')
            // Wrap bullet points in ul tags
            .replace(/(<li>.*?<\/li>(\s*<li>.*?<\/li>)*)/gs, '<ul>$1</ul>')
            // Paragraph breaks
            .replace(/\n\n/g, '<br><br>')
            // Single line breaks
            .replace(/\n/g, '<br>');
    }
    
    // Send message to the API
    async function sendMessageToAPI(message) {
        typingIndicator.style.display = 'block';
        
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
    newChatBtn.addEventListener('click', createNewConversation);
    
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
});