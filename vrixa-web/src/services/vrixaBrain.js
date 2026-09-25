class VrixaBrain {
  constructor() {
    this.responses = {
      'hi': ['Hello! I am ready to assist you.', 'Hi! I am standing by for your command.'],
      'greetings': ['Hello! All systems are operational.'],
      'how are you': ['I am operating at peak efficiency, thank you for asking.'],
      'hello': ['Hello! I am ready to assist you.'],
      'thank you vrixa': ['Always a pleasure to assist you!'],
      'thank you': ['Always a pleasure to assist you!'],
      'introduce': ['I am Vrixa, your personal AI assistant.'],
      'who created you': ['I am Vrixa, your intelligent AI assistant.'],
      'owner': ['The owner and author of Vrixa AI is Harsh.'],
      'author': ['The author of Vrixa AI is Harsh.'],
      'creator': ['Vrixa AI was created by Harsh.'],
      'friend': ['Your close friends are:\n• Harshit\n• Ayush\n• Kartikey\n• Kartik\n• Aniket\n• Bhupender\n\nAlways ready to assist you and your friends!'],
      'friends': ['Your close friends are:\n• Harshit\n• Ayush\n• Kartikey\n• Kartik\n• Aniket\n• Bhupender\n\nAlways ready to assist you and your friends!'],
      'dost': ['Your close friends are:\n• Harshit\n• Ayush\n• Kartikey\n• Kartik\n• Aniket\n• Bhupender\n\nAlways ready to assist you and your friends!'],
    };
  }

  async processInput(userInput) {
    const text = userInput.trim().toLowerCase();
    for (const [key, responses] of Object.entries(this.responses)) {
      if (text.includes(key)) {
        return responses[Math.floor(Math.random() * responses.length)];
      }
    }
    return "I am ready to help. Please tell me what you need.";
  }
}


export const vrixaBrain = new VrixaBrain();
