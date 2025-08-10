import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

class AnonymityManager:
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'name': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Simple name pattern
            'address': r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b'
        }
    
    def sanitize_input(self, input_text):
        """Improved sanitization with better regex patterns"""
        sanitized = input_text
        
        # Replace sensitive information
        sanitized = re.sub(self.patterns['email'], '[EMAIL]', sanitized)
        sanitized = re.sub(self.patterns['phone'], '[PHONE]', sanitized)
        sanitized = re.sub(self.patterns['ssn'], '[SSN]', sanitized)
        sanitized = re.sub(self.patterns['name'], '[NAME]', sanitized)
        sanitized = re.sub(self.patterns['address'], '[ADDRESS]', sanitized)
        
        return sanitized
    
    def is_safe_to_process(self, input_text):
        """Check if input contains sensitive information"""
        for pattern_name, pattern in self.patterns.items():
            if re.search(pattern, input_text):
                return False, f"Detected {pattern_name}"
        return True, "Safe"

# Training data with categories - expanded and balanced
training_data = [
    # Anxious category
    ("I feel anxious about my exams", "anxious"),
    ("I'm having a panic attack", "anxious"),
    ("My heart is racing", "anxious"),
    ("I can't calm myself down", "anxious"),
    ("I'm scared", "anxious"),
    ("I feel scared and lost", "anxious"),
    ("I'm terrified", "anxious"),
    ("I'm not mentally well", "anxious"),
    ("I feel jittery and overwhelmed", "anxious"),
    ("Everything makes me nervous", "anxious"),
    ("I'm panicking right now", "anxious"),
    ("I feel unsafe", "anxious"),
    ("I feel worried all the time", "anxious"),
    ("My mind won't stop racing", "anxious"),
    ("I have butterflies in my stomach", "anxious"),
    ("I feel restless", "anxious"),
    ("I'm having trouble breathing", "anxious"),
    ("I feel on edge", "anxious"),
    ("I'm shaking", "anxious"),
    ("I feel like something bad will happen", "anxious"),
    ("anxious", "anxious"),
    ("panic", "anxious"),
    ("worried", "anxious"),
    ("nervous", "anxious"),
    ("scared", "anxious"),
    ("terrified", "anxious"),

    # Depressed category
    ("I feel hopeless", "depressed"),
    ("Nothing makes me happy anymore", "depressed"),
    ("I'm tired of everything", "depressed"),
    ("I feel empty and lost", "depressed"),
    ("I just want to sleep forever", "depressed"),
    ("I'm not okay", "depressed"),
    ("I feel broken", "depressed"),
    ("I don't feel like doing anything", "depressed"),
    ("I feel like shutting down", "depressed"),
    ("I'm done with life", "depressed"),
    ("I feel like I'm sinking", "depressed"),
    ("I can't get out of bed", "depressed"),
    ("Every day feels pointless", "depressed"),
    ("I feel like a burden", "depressed"),
    ("Nothing interests me", "depressed"),
    ("I feel worthless", "depressed"),
    ("I have no motivation", "depressed"),
    ("Life feels meaningless", "depressed"),
    ("I feel numb inside", "depressed"),
    ("I can't find joy in anything", "depressed"),
    ("I feel like giving up", "depressed"),
    ("Everything feels dark", "depressed"),
    ("depressed", "depressed"),
    ("hopeless", "depressed"),
    ("worthless", "depressed"),
    ("empty", "depressed"),

    # Lonely category
    ("I have no one to talk to", "lonely"),
    ("I feel like nobody cares", "lonely"),
    ("Even in a crowd, I feel alone", "lonely"),
    ("I'm always alone", "lonely"),
    ("No one understands me", "lonely"),
    ("I miss having someone around", "lonely"),
    ("I feel disconnected", "lonely"),
    ("I'm so lonely", "lonely"),
    ("I just want someone to listen", "lonely"),
    ("I need companionship", "lonely"),
    ("I wish I had someone close", "lonely"),
    ("Nobody checks in on me", "lonely"),
    ("I spend every day alone", "lonely"),
    ("I feel isolated", "lonely"),
    ("I have no friends", "lonely"),
    ("Everyone has left me", "lonely"),
    ("I feel abandoned", "lonely"),
    ("No one wants to be around me", "lonely"),
    ("I'm all by myself", "lonely"),
    ("lonely", "lonely"),
    ("alone", "lonely"),
    ("isolated", "lonely"),
    ("abandoned", "lonely"),

    # Sad category
    ("I've been crying a lot", "sad"),
    ("I'm so sad today", "sad"),
    ("I feel really down", "sad"),
    ("My mood is low", "sad"),
    ("I can't stop the tears", "sad"),
    ("Everything feels heavy", "sad"),
    ("Nothing feels right", "sad"),
    ("I'm not feeling good", "sad"),
    ("Today I just feel off", "sad"),
    ("I feel like crying", "sad"),
    ("I feel like everything's going wrong", "sad"),
    ("I've lost my spark", "sad"),
    ("I feel blue", "sad"),
    ("My heart feels heavy", "sad"),
    ("I'm feeling down", "sad"),
    ("I'm having a bad day", "sad"),
    ("I feel gloomy", "sad"),
    ("Everything makes me want to cry", "sad"),
    ("sad", "sad"),
    ("crying", "sad"),
    ("tears", "sad"),
    ("blue", "sad"),
    ("down", "sad"),

    # Tired category
    ("I'm mentally exhausted", "tired"),
    ("Even after sleep, I'm tired", "tired"),
    ("I have no energy", "tired"),
    ("I just want to lie down all day", "tired"),
    ("I'm burned out", "tired"),
    ("I feel completely drained", "tired"),
    ("I'm overwhelmed and tired", "tired"),
    ("I can't focus anymore", "tired"),
    ("I feel numb", "tired"),
    ("Everything takes effort", "tired"),
    ("I'm so fatigued", "tired"),
    ("I feel worn out", "tired"),
    ("I'm exhausted all the time", "tired"),
    ("I can barely keep my eyes open", "tired"),
    ("I'm running on empty", "tired"),
    ("tired", "tired"),
    ("exhausted", "tired"),
    ("drained", "tired"),
    ("fatigued", "tired"),
    ("burnout", "tired"),

    # Stress category
    ("I'm overwhelmed with work", "stress"),
    ("Too many deadlines are stressing me out", "stress"),
    ("I feel like I can't handle this pressure", "stress"),
    ("Everything is stressing me out", "stress"),
    ("I have too much on my plate", "stress"),
    ("Deadlines are killing me", "stress"),
    ("My brain won't slow down", "stress"),
    ("I can't relax", "stress"),
    ("I'm constantly under pressure", "stress"),
    ("I'm snapping at everyone", "stress"),
    ("I feel overwhelmed", "stress"),
    ("There's too much to do", "stress"),
    ("I'm falling behind", "stress"),
    ("I can't keep up", "stress"),
    ("Everything is urgent", "stress"),
    ("I'm at my breaking point", "stress"),
    ("stress", "stress"),
    ("overwhelmed", "stress"),
    ("pressure", "stress"),
    ("deadlines", "stress"),

    # Happy category
    ("I feel great today", "happy"),
    ("I'm so happy", "happy"),
    ("Everything is going well", "happy"),
    ("I feel amazing", "happy"),
    ("I'm having a wonderful day", "happy"),
    ("I feel joyful", "happy"),
    ("Life is good", "happy"),
    ("I'm feeling fantastic", "happy"),
    ("I'm in a great mood", "happy"),
    ("I feel blessed", "happy"),
    ("happy", "happy"),
    ("joyful", "happy"),
    ("excited", "happy"),
    ("wonderful", "happy"),
    ("fantastic", "happy"),

    # Angry category
    ("I'm so angry", "angry"),
    ("I'm furious", "angry"),
    ("I'm mad about this", "angry"),
    ("I'm frustrated", "angry"),
    ("This makes me angry", "angry"),
    ("I'm irritated", "angry"),
    ("I'm pissed off", "angry"),
    ("I'm annoyed", "angry"),
    ("I can't stand this", "angry"),
    ("This is infuriating", "angry"),
    ("angry", "angry"),
    ("furious", "angry"),
    ("mad", "angry"),
    ("frustrated", "angry"),
    ("irritated", "angry"),

    # Confused category
    ("I don't understand", "confused"),
    ("I'm confused", "confused"),
    ("I don't know what to do", "confused"),
    ("I'm lost", "confused"),
    ("I'm uncertain", "confused"),
    ("I don't get it", "confused"),
    ("I'm puzzled", "confused"),
    ("I'm bewildered", "confused"),
    ("I'm not sure", "confused"),
    ("I'm mixed up", "confused"),
    ("confused", "confused"),
    ("uncertain", "confused"),
    ("puzzled", "confused"),
    ("bewildered", "confused"),
    ("lost", "confused"),

    # General category - expanded to balance dataset
    ("hello", "general"),
    ("hi", "general"),
    ("hey", "general"),
    ("good morning", "general"),
    ("how are you?", "general"),
    ("thank you", "general"),
    ("you there?", "general"),
    ("can we talk?", "general"),
    ("i need to chat", "general"),
    ("just saying hi", "general"),
    ("you're kind", "general"),
    ("thanks for listening", "general"),
    ("what can you do?", "general"),
    ("do you understand me?", "general"),
    ("i need help", "general"),
    ("i just want to talk", "general"),
    ("let's chat", "general"),
    ("good evening", "general"),
    ("what's up", "general"),
    ("how's it going", "general"),
    ("nice to meet you", "general"),
    ("tell me about yourself", "general"),
    ("are you a robot", "general"),
    ("can you help me", "general"),
    ("i'm here", "general"),
    ("good night", "general"),
    ("see you later", "general"),
    ("goodbye", "general"),
    ("take care", "general"),
    ("thanks", "general"),
    ("okay", "general"),
    ("alright", "general"),
    ("cool", "general"),
    ("sure", "general"),
    ("yes", "general"),
    ("no", "general"),
    ("maybe", "general"),
    ("i don't know", "general"),
    ("whatever", "general"),
    ("fine", "general"),
    ("hmm", "general"),
    ("interesting", "general"),
    ("really", "general"),
    ("wow", "general"),
    ("oh", "general"),
    ("ah", "general"),
    ("well", "general"),
    ("so", "general"),
    ("anyway", "general"),
    ("actually", "general"),
    ("basically", "general"),
    ("generally", "general"),
    ("usually", "general"),
    ("sometimes", "general"),
    ("always", "general"),
    ("never", "general"),
    ("i'm fine", "general"),
    ("everything is good", "general"),
    ("life is okay", "general"),
    ("nothing much", "general"),
    ("just chilling", "general"),
    ("having a normal day", "general"),
    ("things are alright", "general"),
    ("feeling neutral", "general"),
    ("it's a regular day", "general"),
    ("everything's normal", "general"),
]

class TherapyChatbot:
    def __init__(self, debug=False):
        self.debug = debug
        self.anonymity_manager = AnonymityManager()
        
        # Prepare training data
        texts, labels = zip(*training_data)
        
        # Improved vectorizer settings
        self.model = make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 2),  # Include bigrams for better context
                max_features=5000,   # Increase feature space
                lowercase=True,
                stop_words=None,     # Keep stop words as they can be meaningful in emotional context
                sublinear_tf=True    # Use sublinear scaling
            ), 
            LogisticRegression(
                max_iter=2000,       # Increase iterations
                C=1.0,               # Regularization parameter
                class_weight='balanced',  # Handle class imbalance
                random_state=42
            )
        )
        
        # Train the model
        self.model.fit(texts, labels)
        self.labels = self.model.classes_
        
        # Evaluate model if debug mode
        if self.debug:
            self.evaluate_model(texts, labels)

        self.responses = {
            "anxious": [
                "It's okay to feel anxious. Try some grounding techniques like deep breathing - inhale for 4, hold for 4, exhale for 4.",
                "Take a moment to slow down and breathe. Anxiety can be managed step by step. You're safe right now.",
                "I hear that you're feeling anxious. Try focusing on 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
                "Anxiety is your body's way of trying to protect you. Let's work through this together. What's one small thing you can do right now to feel more grounded?",
                "I understand you're feeling anxious. Remember that anxiety is temporary, and you have the strength to get through this moment."
            ],
            "depressed": [
                "You matter, and I'm really sorry you're feeling this way. Depression can feel overwhelming, but you're not alone.",
                "These feelings are valid, and it's okay to not be okay. Would you like to talk more about what's going on?",
                "Depression can make everything feel hopeless, but these feelings will pass. You're stronger than you know.",
                "I'm here to listen without judgment. Sometimes just sharing these feelings can help lighten the load a little.",
                "I hear the pain in your words. Please know that you're worthy of care and support, even when it doesn't feel that way."
            ],
            "lonely": [
                "Feeling lonely is really tough. Talking here is a good first step, and I'm glad you reached out.",
                "Loneliness can feel overwhelming, but remember that connection is possible. You're not as alone as you feel right now.",
                "I hear you, and your feelings are valid. Sometimes reaching out to one person, even online, can help break the isolation.",
                "You deserve companionship and understanding. While I'm just a chatbot, I'm here to listen and remind you that you matter.",
                "Loneliness is painful, but it's also a sign that you value connection. That's a beautiful part of who you are."
            ],
            "sad": [
                "Sadness is a natural human emotion. It's okay to feel this way, and I'm here with you through it.",
                "It's okay to cry or feel down. Want to share what's making you feel this way? Sometimes talking helps.",
                "Your feelings are valid. Sadness shows that you care deeply about something. That's actually a beautiful part of being human.",
                "I'm sorry you're going through this difficult time. Remember that feelings, even heavy ones, are temporary.",
                "I can hear the sadness in your words. It's okay to sit with these feelings - they're part of your human experience."
            ],
            "tired": [
                "It sounds like you're mentally and emotionally exhausted. Rest is not just important, it's necessary.",
                "Burnout is real, and it's your body and mind telling you to slow down. Be gentle with yourself.",
                "When everything feels draining, sometimes the most productive thing you can do is rest. Your energy matters.",
                "Mental exhaustion is just as real as physical exhaustion. Take breaks when you need them - you deserve care.",
                "I hear how tired you are. It's okay to not be at 100% all the time. Your wellbeing comes first."
            ],
            "stress": [
                "Stress can really wear you down. Try to break things down into smaller, manageable parts.",
                "You've been dealing with a lot. It's okay to pause, breathe, and reset. You don't have to handle everything at once.",
                "When everything feels urgent, sometimes stepping back can help you see what truly needs attention first.",
                "Stress is your mind's way of saying 'this is a lot.' Listen to that signal and be kind to yourself.",
                "I can feel the pressure you're under. Remember that you can only do what you can do, and that's enough."
            ],
            "happy": [
                "That's wonderful to hear! I'm so glad you're feeling happy. What's bringing you joy today?",
                "It's beautiful when happiness shines through! Your positive energy is contagious. Care to share what's making you feel so good?",
                "I love hearing that you're in a good place! Happiness is such a gift. What's been going well for you?",
                "Your happiness brightens my day too! It's amazing how good feelings can be so uplifting. What's been the highlight of your day?",
                "I'm so happy to hear you're feeling good! These moments of joy are precious - savor them."
            ],
            "confused": [
                "It's completely normal to feel confused sometimes. Life can be complex and uncertain. What's been on your mind?",
                "Uncertainty can be uncomfortable, but it's also a sign that you're thinking deeply about things. Want to talk through what's confusing you?",
                "Feeling lost or uncertain is part of being human. Sometimes talking through our confusion can help bring clarity. I'm here to listen.",
                "It's okay not to have all the answers. Confusion often comes before understanding. What's been weighing on your mind?",
                "I hear that you're feeling uncertain. That's a natural part of growth and learning. What's been puzzling you?"
            ],
            "angry": [
                "I can hear that you're feeling really frustrated. Anger is a valid emotion - it often tells us something important needs attention.",
                "It sounds like you're dealing with something really frustrating. Want to talk about what's making you feel this way?",
                "Anger can be overwhelming, but it's also a signal that something matters to you. I'm here to listen without judgment.",
                "I understand you're feeling angry. Sometimes expressing these feelings in a safe space can help. What's been bothering you?",
                "Your anger is valid. It's telling you that something doesn't feel right. What's been triggering these feelings?"
            ],
            "general": [
                "Hi there! I'm here to listen whenever you're ready to share what's on your mind.",
                "Hello! How are you feeling today? I'm here if you need someone to talk to.",
                "Feel free to share whatever's on your mind. I'm listening and here to support you.",
                "Welcome! I'm here to provide a safe space for you to express your thoughts and feelings.",
                "I'm glad you're here. What would you like to talk about today?"
            ]
        }

        # Crisis keywords that need immediate attention
        self.crisis_keywords = [
            'suicide', 'suicidal', 'kill myself', 'end it all', 'hurt myself', 
            'self harm', 'cutting', 'die', 'death', 'overdose', 'pills',
            'kill myself', 'want to die', 'better off dead', 'no reason to live',
            'take my own life', 'harm myself', 'end it all', 'can\'t go on',
            'ready to give up', 'no point in living', 'planning to hurt myself'
        ]

    def evaluate_model(self, texts, labels):
        """Evaluate model performance"""
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Retrain on training split
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        print("Model Performance:")
        print(classification_report(y_test, y_pred))
        print(f"Classes: {self.labels}")

    def preprocess_message(self, message):
        """Clean and preprocess the message for better classification"""
        # Convert to lowercase
        message = message.lower().strip()
        
        # Handle common contractions
        contractions = {
            "i'm": "i am",
            "i've": "i have",
            "i'll": "i will",
            "i'd": "i would",
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not"
        }
        
        for contraction, expansion in contractions.items():
            message = message.replace(contraction, expansion)
        
        return message

    def detect_crisis(self, user_input):
        """Detect if user input contains crisis-related keywords"""
        text_lower = user_input.lower()
        return any(keyword in text_lower for keyword in self.crisis_keywords)
    
    def get_crisis_response(self):
        """Return crisis response with resources"""
        return """🚨 I'm concerned about what you've shared. If you're having thoughts of suicide or self-harm, please reach out for help immediately:

• National Suicide Prevention Lifeline: 988 (US)
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911
• International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

You matter, and help is available. Please consider reaching out to a mental health professional or trusted person in your life."""

    def generate_response(self, message):
        """Generate appropriate response based on message analysis"""
        # Check for crisis first
        if self.detect_crisis(message):
            return self.get_crisis_response()
            
        # Check for sensitive information
        is_safe, reason = self.anonymity_manager.is_safe_to_process(message)
        if not is_safe:
            return f"I notice you've shared some personal information. For your privacy, I'd prefer if you could share your feelings without including specific details like {reason.split()[1]}. How are you feeling right now?"
            
        # Sanitize and preprocess the message
        sanitized_message = self.anonymity_manager.sanitize_input(message)
        processed_message = self.preprocess_message(sanitized_message)
        
        # Handle very short messages
        if len(processed_message.strip()) == 0:
            return random.choice(self.responses["general"])
        
        # Get prediction probabilities
        probs = self.model.predict_proba([processed_message])[0]
        predicted = self.model.predict([processed_message])[0]
        confidence = max(probs)
        
        # Get the top 3 predictions for debugging
        if self.debug:
            prob_dict = dict(zip(self.labels, probs))
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            
            print(f"[Intent Analysis] Top predictions:")
            for i, (label, prob) in enumerate(sorted_probs[:3]):
                print(f"  {i+1}. {label}: {prob:.3f}")
            print(f"[Final Decision] {predicted} (confidence: {confidence:.3f})")
        
        # Improved confidence handling
        if confidence < 0.25:
            return "I want to make sure I understand you correctly. Can you tell me more about what you're experiencing or feeling right now?"
        elif confidence < 0.35 and predicted == "general":
            return "I'm here to listen. Can you share more about what's on your mind or how you're feeling?"
        elif confidence < 0.40 and predicted in ["confused", "general"]:
            return "I want to make sure I understand you. Could you share a bit more about what you're going through?"
        
        return random.choice(self.responses.get(predicted, self.responses["general"]))
    
    def get_response(self, user_input):
        """Main method to get response from the chatbot"""
        if not user_input or not user_input.strip():
            return "I'm here to listen. What would you like to talk about?"
        
        return self.generate_response(user_input)

    def interactive_session(self):
        """Start an interactive chat session"""
        print("🤖 Therapy Chatbot - I'm here to listen and support you.")
        print("Type 'quit' to exit the conversation.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Chatbot: Take care of yourself. Remember, you're not alone. 💙")
                    break
                
                response = self.get_response(user_input)
                print(f"Chatbot: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nChatbot: Take care of yourself. Remember, you're not alone. 💙")
                break
            except Exception as e:
                print(f"Chatbot: I'm sorry, I encountered an error. Can you try rephrasing that? Error: {str(e)}\n")


def main():
    """Main function to run the chatbot"""
    print("Initializing Therapy Chatbot...")
    chatbot = TherapyChatbot(debug=True)
    
    print("\nTesting the chatbot with sample inputs:")
    test_inputs = [
        "I feel anxious about everything",
        "I'm stressed about work deadlines",
        "I feel so depressed and hopeless",
        "I'm really happy today!",
        "I'm confused about my life",
        "I'm angry at my situation",
        "Hello there",
        "Thank you for listening"
    ]
    
    for test_input in test_inputs:
        response = chatbot.get_response(test_input)
        print(f"\nInput: {test_input}")
        print(f"Response: {response}")
        print("-" * 60)
    
    # Start interactive session
    print("\n" + "="*60)
    print("Starting interactive session...")
    chatbot.interactive_session()


if __name__ == "__main__":
    main()