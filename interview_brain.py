import os
import uuid
import random
from groq import Groq
from dotenv import load_dotenv

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class MockInterviewBrain:
    def __init__(self):
        """Initialize the Groq client."""
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        self.client = Groq(api_key=GROQ_API_KEY)
        self.sessions = {}  # In-memory storage for active interviews

    def create_session(self, resume_text, job_role):
        """Creates a new interview session."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "resume_text": resume_text,
            "target_role": job_role,
            "history": []
        }
        print(f"DEBUG: New Session Created: {session_id}")
        return session_id

    def get_next_question(self, session_id):
        """
        Logic Flow:
        - Turn 0: Hard-coded Greeting.
        - Turn 1-8: AI Questions (With "Nuclear" Safety Net).
        - Turn 9: Feedback.
        """
        session = self.sessions.get(session_id)
        if not session:
            return {"question": "Error: Session lost. Please restart."}

        # 1. COUNT TURNS
        ai_turns = len([m for m in session['history'] if m['role'] == 'assistant'])
        print(f"DEBUG: Session {session_id} | Turn: {ai_turns}")

        # --- STEP 1: HARD-CODED START (Turn 0) ---
        if ai_turns == 0:
            question = "Hi! I've seen your resume. Let's start with a quick introduction about yourself."
            session['history'].append({"role": "assistant", "content": question})
            return {"question": question}

        # --- STEP 2: DECIDE PHASE ---
        if ai_turns < 9:
            phase = "QUESTION"
        else:
            phase = "FEEDBACK"

        # 2. PREPARE CONTEXT (Last answer only)
        # We limit context so the AI doesn't get confused by the whole history
        last_user_msg = session['history'][-1]['content'] if session['history'] else "No answer"

        # 3. CONSTRUCT THE PROMPT
        if phase == "QUESTION":
            prompt = f"""
            You are a Technical Interviewer.
            Current Question Number: {ai_turns} of 9.
            
            The Candidate just said: "{last_user_msg}"
            
            YOUR TASK:
            1. Ask a follow-up technical question based on their answer.
            2. If their answer was short/vague, ask a fundamental question about {session['target_role']}.
            
            STRICT RULES:
            - **NO GREETINGS:** Do not say "Hi", "Hello", "Great", "Okay". Just the question.
            - **LENGTH:** Max 1 sentence. Short and crisp.
            - **FORBIDDEN:** DO NOT say "Interview Complete". You must ask a question.
            """
            
        elif phase == "FEEDBACK":
            # Build transcript so the AI knows what happened
            transcript = ""
            for msg in session['history']:
                role_title = "Interviewer" if msg['role'] == "assistant" else "Candidate"
                transcript += f"{role_title}: {msg['content']}\n"

            prompt = f"""
            The interview is over.
            Role: {session['target_role']}
            
            Transcript:
            {transcript}
            
            Task: Rate the candidate out of 10 and give 1 sentence of feedback based on the transcript.
            Format: "Score: X/10. [Feedback]. Interview Complete."
            Example: "Score: 7/10. Good conceptual knowledge but work on SQL. Interview Complete."
            """

        try:
            # 4. GENERATE RESPONSE
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.6 
            )
            question = response.choices[0].message.content.strip()
            
            # 5. NUCLEAR SAFETY NET (The Fix)
            # If the AI ignored us and tried to end, we DELETE its response and force a new one.
            lower_q = question.lower()
            if phase == "QUESTION" and ("interview complete" in lower_q or "thank you" in lower_q or len(question) < 5):
                print("DEBUG: AI tried to quit. activating Safety Net.")
                
                # List of backup questions to ensure the interview continues
                backups = [
                    "What is your preferred programming language and why?",
                    "Can you explain the difference between a List and a Tuple?",
                    "How do you handle errors or exceptions in your code?",
                    "Describe a challenging bug you fixed recently.",
                    "What are the key principles of Object-Oriented Programming?",
                    "How do you optimize a slow database query?"
                ]
                # Pick a random one
                question = random.choice(backups)

            # 6. CLEANUP (Remove "Question:" prefix if AI adds it)
            question = question.replace("Question:", "").strip()

            session['history'].append({"role": "assistant", "content": question})
            return {"question": question}
            
        except Exception as e:
            print(f"ERROR: {e}")
            return {"question": "Let's move on. What is your strongest technical skill?"}

    def answer_question(self, session_id, answer):
        """Stores the user's answer in history."""
        session = self.sessions.get(session_id)
        if session:
            session['history'].append({"role": "user", "content": answer})
            return {"status": "Answer recorded"}
        return {"error": "Session not found"}
