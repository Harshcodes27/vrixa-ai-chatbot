import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ChatArea from './components/ChatArea';
import MessageInput from './components/MessageInput';
import SettingsModal from './components/SettingsModal';
import { vrixaBrain } from './services/vrixaBrain';
import { speechService } from './services/speechService';

const INITIAL_GREETING = 'Welcome to Vrixa AI! Please tell me your name and how I can help you.';
const VRIXA_INTRODUCTION = 'Vrixa AI is an intelligent chatbot designed to answer questions, assist with tasks, and provide helpful conversations.';

function createSession(id) {
  return {
    id,
    title: 'New Conversation',
    messages: [
      { sender: 'bot', text: INITIAL_GREETING, timestamp: new Date().toISOString() },
      { sender: 'bot', text: VRIXA_INTRODUCTION, timestamp: new Date().toISOString() }
    ]
  };
}

function normalizeSession(session) {
  const messages = Array.isArray(session.messages) ? session.messages : [];
  const firstMessage = messages[0];
  const isLegacyOnboarding = firstMessage?.sender === 'bot' && (
    firstMessage.text === 'Hi! What’s your name?' ||
    firstMessage.text === "Hi! What's your name?" ||
    firstMessage.text === 'Hello Sir, standing by for instructions.'
  );
  if (isLegacyOnboarding) return createSession(session.id);

  const hasUserMessage = messages.some(message => message.sender === 'user');
  return hasUserMessage ? session : createSession(session.id);
}

function isGreeting(text) {
  return /^(?:hi|hello|hey|hii|heyy|hlo|hlw|namaste|hola|sup|yo|vrixa)(?:\s+vrixa)?[.!?]*$/i.test(text.trim());
}

function extractName(text) {
  const explicitName = text.match(/^(?:my name is|i am|i'm)\s+(.+?)[.!?]?$/i);
  const candidate = explicitName ? explicitName[1] : text;
  const name = candidate.trim().replace(/[.!?]+$/, '').trim();
  const isSimpleName = /^[A-Za-z][A-Za-z .'-]{0,49}$/.test(name);
  const isBlockedPhrase = /^(?:hi|hello|hey|hii|namaste|who am i|what is your name|help|help me)$/i.test(name);
  const isQuestionOrConversation = /^(?:who|what|how|why|when|where|can|do|does|is|are|tell|please|tujhe|tumhe|kisne|kaise|kya|nice to meet you)\b/i.test(name);

  return isSimpleName && !isBlockedPhrase && !isQuestionOrConversation ? name : null;
}

export default function App() {
  const [sessions, setSessions] = useState(() => {
    const saved = localStorage.getItem('VRIXA_CHAT_SESSIONS');
    if (saved) {
      try {
        return JSON.parse(saved).map(normalizeSession);
      } catch (e) { console.error(e); }
    }
    return [createSession('default-1')];
  });

  const [activeSessionId, setActiveSessionId] = useState(() => {
    return localStorage.getItem('VRIXA_ACTIVE_SESSION_ID') || 'default-1';
  });

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [isThinking, setIsThinking] = useState(false);

  useEffect(() => {
    localStorage.setItem('VRIXA_CHAT_SESSIONS', JSON.stringify(sessions));
  }, [sessions]);

  useEffect(() => {
    localStorage.setItem('VRIXA_ACTIVE_SESSION_ID', activeSessionId);
  }, [activeSessionId]);

  const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];

  const handleNewChat = () => {
    const newId = 'chat-' + Date.now();
    const newSession = createSession(newId);
    setSessions(prev => [newSession, ...prev]);
    setActiveSessionId(newId);
    setSidebarOpen(false);
  };

  const handleSelectSession = (id) => {
    setActiveSessionId(id);
    setSidebarOpen(false);
  };

  const handleDeleteSession = (id) => {
    setSessions(prev => {
      const filtered = prev.filter(s => s.id !== id);
      if (filtered.length === 0) {
        const fallback = createSession('chat-' + Date.now());
        setActiveSessionId(fallback.id);
        return [fallback];
      }
      if (activeSessionId === id) {
        setActiveSessionId(filtered[0].id);
      }
      return filtered;
    });
  };

  const handleSendMessage = async (text, attachments = []) => {
    if (!text.trim() && attachments.length === 0) return;

    const userMsg = { sender: 'user', text, attachments, timestamp: new Date().toISOString() };

    setSessions(prev => prev.map(session => {
      if (session.id === activeSessionId) {
        const updatedMsgs = [...session.messages, userMsg];
        const updatedTitle = session.messages.length === 0 
          ? (text ? (text.length > 25 ? text.substring(0, 25) + '...' : text) : 'Attachment Analysis') 
          : session.title;
        return { ...session, title: updatedTitle, messages: updatedMsgs };
      }
      return session;
    }));

    setIsThinking(true);

    try {
      const isAwaitingName = activeSession?.messages.some(message => (
        message.sender === 'bot' && message.text === INITIAL_GREETING
      )) && !activeSession.messages.some(message => message.sender === 'user');
      const name = isAwaitingName && attachments.length === 0 ? extractName(text) : null;
      if (name) {
        setSessions(prev => prev.map(session => (
          session.id === activeSessionId ? { ...session, userName: name } : session
        )));
      }
      const replyText = name
        ? `Nice to meet you, ${name}! How can I help you today?`
        : isAwaitingName
          ? INITIAL_GREETING
          : activeSession?.userName && attachments.length === 0 && isGreeting(text)
            ? `Hello ${activeSession.userName}! How can I help you today?`
          : await vrixaBrain.processInput(text, attachments);
      const botMsg = { sender: 'bot', text: replyText, timestamp: new Date().toISOString() };

      setSessions(prev => prev.map(session => {
        if (session.id === activeSessionId) {
          return { ...session, messages: [...session.messages, botMsg] };
        }
        return session;
      }));

      if (ttsEnabled) {
        speechService.speak(replyText);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsThinking(false);
    }
  };

  return (
    <div className="app-container">
      <Sidebar 
        isOpen={sidebarOpen} 
        closeSidebar={() => setSidebarOpen(false)}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
      />

      <div className="main-wrapper">
        <Header 
          toggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          openSettings={() => setSettingsOpen(true)}
          ttsEnabled={ttsEnabled}
          setTtsEnabled={setTtsEnabled}
        />

        <ChatArea 
          messages={activeSession ? activeSession.messages : []}
          onSelectStarter={handleSendMessage}
          isThinking={isThinking}
        />

        <MessageInput 
          onSendMessage={handleSendMessage}
          disabled={isThinking}
        />
      </div>

      <SettingsModal 
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  );
}
