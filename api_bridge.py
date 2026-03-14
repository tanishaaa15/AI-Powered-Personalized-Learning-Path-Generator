from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv
import json
import uuid
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    print("✅ Server Running! Endpoints: /transcribe_audio available.")

# --- SETUP RAG (Skill Gap & Context) ---
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    print("✅ O*NET Database Ready!")
except Exception as e:
    print(f"⚠️ RAG Database Warning: {e}")
    vector_db = None

# --- MISSING CLASSES (The Fix) ---
class ResumeRequest(BaseModel):
    resume_text: str
    target_role: str = "Professional"

class RoadmapRequest(BaseModel):
    resume_text: str
    target_role: str
    weeks: int = 4
    daily_hours: int = 2
    language: str = "English"

class InterviewStartRequest(BaseModel):
    resume_text: str
    target_role: str

class AnswerRequest(BaseModel):
    session_id: str
    answer: str

class ProfileRequest(BaseModel):
    resume_text: str

# --- 1. SKILL GAP ANALYSIS ---
# --- REPLACE THIS FUNCTION IN api_bridge 1.py ---

# --- REPLACE THIS FUNCTION IN api_bridge 1.py ---

@app.post("/analyze_skills")
def analyze_skills(request: ResumeRequest):
    # 1. PYTHON DECIDES THE ROLE (Logic > AI)
    role = "Professional" # Default fallback
    
    if request.target_role and request.target_role.strip():
        # Case A: User typed something
        role = request.target_role
        print(f"🎯 User-Defined Role: {role}")
    else:
        # Case B: Auto-Detect (The AI Guess)
        print("🕵️ Auto-detecting role from resume...")
        detect_prompt = f"""
        Read the resume below and identify the SINGLE most likely Job Role this candidate is targeting (e.g., "Data Scientist", "Java Developer", "Digital Marketer").
        Return ONLY the role name. Do not add punctuation. If unsure, return "General Professional".
        
        RESUME:
        {request.resume_text[:1500]}
        """
        try:
            # We use a separate cheap call just for detection
            role_response = client.chat.completions.create(
                messages=[{"role": "user", "content": detect_prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1
            )
            detected = role_response.choices[0].message.content.strip()
            # Clean up potential mess (remove "The role is ")
            if len(detected) < 50: 
                role = detected
            print(f"✅ Auto-Detected: {role}")
        except Exception as e:
            print(f"⚠️ Detection Failed: {e}")
            role = "General Professional"

    # 2. RAG RETRIEVAL
    context = ""
    if vector_db:
        try:
            results = vector_db.similarity_search(role, k=3)
            context = "\n".join([doc.page_content for doc in results])
        except: pass

    # 3. ANALYSIS PROMPT
    prompt = f"""
    Act as a strict Technical Recruiter.
    Compare the Candidate's Resume against the Official Industry Standards for the role: '{role}'.
    
    CANDIDATE RESUME:
    {request.resume_text[:3000]}
    
    OFFICIAL INDUSTRY STANDARDS:
    {context}
    
    INSTRUCTIONS:
    1. Carefully read the 'CANDIDATE RESUME'.
    2. 'present_skills': Extract ONLY the specific hard technical skills (languages, frameworks, libraries, tools, or databases) that are EXPLICITLY mentioned in the resume and are RELEVANT to the role of '{role}'.
       - Do NOT include generic phrases (e.g., 'Web Development', 'Problem Solving').
       - Do NOT include soft skills (e.g., 'Leadership', 'Communication').
       - If a skill is not explicitly written in the text, do NOT include it.
    3. 'missing_skills': Identify the top 5-10 CRITICAL technical skills required for '{role}' (based on 'OFFICIAL INDUSTRY STANDARDS') that are NOT found in the resume.
       - Do NOT list skills that are already in 'present_skills'.
       - Focus on specific, high-demand frameworks, languages, and tools.
    
    RETURN JSON ONLY:
    {{
        "present_skills": ["Skill1", "Skill2"],
        "missing_skills": ["Skill3", "Skill4"]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", 
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        
        # --- PYTHON CALCULATION FOR FAIRNESS ---
        # Calculate score strictly based on the visible skills (Verified vs Gaps)
        p_count = len(data.get("present_skills", []))
        m_count = len(data.get("missing_skills", []))
        total = p_count + m_count
        
        # Avoid division by zero. Round to nearest integer.
        data["match_percentage"] = int(round((p_count / total) * 100)) if total > 0 else 0
        
        # --- THE FIX: PYTHON FORCES THE ROLE NAME ---
        # We manually overwrite the 'role' key so it matches exactly what we detected.
        # This prevents the AI from returning "None".
        data["role"] = role 
        
        return data
        
    except Exception as e:
        print(f"❌ Skill Gap Error: {e}")
        return {"error": str(e)}
    
# --- 2. GENERATE ROADMAP (The Robust "Self-Healing" Version) ---
@app.post("/generate_roadmap")
def generate_roadmap(request: RoadmapRequest):
    print(f"🗺️ Generating Roadmap for: {request.target_role} in {request.language} ({request.weeks} Weeks)")

    # 1. RAG RETRIEVAL
    context_text = "Standard Industry Requirements"
    if vector_db:
        try:
            results = vector_db.similarity_search(request.target_role, k=3)
            context_text = "\n".join([doc.page_content[:500] for doc in results])
        except Exception:
            print("   -> Retrieval Error (Non-critical): Continuing without RAG context.")

    # 2. STRICT SYSTEM PROMPT
    # We force the AI to adopt a persona that REFUSES to speak English for content.
    system_instruction = f"""
    You are a strict Career Curriculum Architect who speaks ONLY {request.language}.
    You are writing a syllabus for a student who does NOT understand English well.
    
    CRITICAL RULES:
    1. OUTPUT JSON ONLY.
    2. 'week_topic', 'day_topic', and 'description' MUST be in {request.language}.
    3. NEVER switch to English for descriptions, even in later weeks.
    4. 'video_search' MUST be in English (for YouTube search accuracy).
    """

    # 3. USER PROMPT
    user_prompt = f"""
    Create a {request.weeks}-WEEK roadmap for '{request.target_role}'.
    
    OFFICIAL CONTEXT:
    {context_text}
    
    STRUCTURE RULES:
    1. Generate EXACTLY {request.weeks} weeks.
    2. Each week MUST have EXACTLY 6 days (Day 1 to Day 6).
    
    REQUIRED JSON FORMAT:
    {{
      "roadmap": [
        {{
          "week_topic": "Topic in {request.language}",
          "recommended_course": "Official Course Name",
          "daily_breakdown": [
             {{ "day_topic": "Topic in {request.language}", "description": "Details in {request.language}", "video_search": "English Search Query", "documentation": "URL" }},
             ... (Repeat for Day 1 to Day 6)
          ]
        }}
        ... (Repeat for all {request.weeks} weeks)
      ]
    }}
    """
    
    try:
        # Using 70b-versatile as per your request
        # (WARNING: If you get Rate Limit Error 429, switch this line to "llama-3.1-8b-instant")
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,  # Keep it strict
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        
        # --- THE "SELF-HEALING" REPAIR LAYER ---
        
        roadmap = data.get("roadmap", [])
        
        # FIX 1: Enforce Week Count (Chop or Pad)
        if len(roadmap) > request.weeks:
            roadmap = roadmap[:request.weeks]
        elif len(roadmap) < request.weeks:
            # If AI gave up early, duplicate the last week to fill the gap
            while len(roadmap) < request.weeks:
                if len(roadmap) > 0:
                    roadmap.append(roadmap[-1].copy()) # Copy last week
                else:
                    break 

        # FIX 2: Enforce 6 Days Per Week (The "Lazy Day" Fix)
        for week in roadmap:
            days = week.get("daily_breakdown", [])
            
            # If AI generated fewer than 6 days, FILL the gaps.
            if len(days) < 6:
                print(f"⚠️ Repairing Week: AI generated only {len(days)} days. Padding to 6.")
                missing_count = 6 - len(days)
                for _ in range(missing_count):
                    # Create a generic "Practice/Revision" day
                    last_day = days[-1] if days else {
                        "day_topic": "Practice", 
                        "description": "Review concepts", 
                        "video_search": f"{request.target_role} tutorial", 
                        "documentation": "#"
                    }
                    days.append(last_day)
            
            # If AI generated too many days, chop them.
            if len(days) > 6:
                week["daily_breakdown"] = days[:6]

        data["roadmap"] = roadmap
        return data
        
    except Exception as e:
        print(f"❌ Roadmap Error: {e}")
        return {"roadmap": [], "error": str(e)}

# --- 3. MOCK INTERVIEW (Fixed Format) ---
from interview_brain import MockInterviewBrain

try:
    interview_bot = MockInterviewBrain()
except Exception as e:
    print(f"⚠️ Brain Error: {e}")
    interview_bot = None

@app.post("/start_interview")
def start_interview(request: InterviewStartRequest):
    if not interview_bot:
        return {"error": "Server Error: Check API Key"}
    
    session_id = interview_bot.create_session(request.resume_text, request.target_role)
    response = interview_bot.get_next_question(session_id)
    
    # FIX: Return nested JSON to match app.py expectation
    return {
        "session_id": session_id,
        "first_question": {
            "question": response.get("question", "Hello! Let's begin.")
        }
    }

@app.post("/answer")
def answer_interview(request: AnswerRequest):
    if not interview_bot:
        return {"next_question": "Server Error."}
    
    interview_bot.answer_question(request.session_id, request.answer)
    response = interview_bot.get_next_question(request.session_id)
    
    return {
        "next_question": response.get("question", "Interview Complete.")
    }
# --- 4. EXTRACT PROFILE (New Feature) ---
@app.post("/extract_profile")
def extract_profile(request: ProfileRequest):
    print("👤 Extracting Profile Details...")

    prompt = f"""
    You are an expert Resume Parser. Your task is to extract specific details from the resume text provided below.
    
    RESUME TEXT:
    {request.resume_text[:4000]}

    INSTRUCTIONS:
    1. "name": Extract ONLY the Candidate's Full Name. 
       - The name is usually the very first line or in the header.
       - CRITICAL: Do NOT include locations, addresses, or city names (e.g., "New York", "Pali, Raj", "India").
       - If the text says "John Doe New York", extract ONLY "John Doe".
    2. "email": (String) Email address.
    3. "phone": (String) Phone number.
    4. "education": Extract the HIGHEST degree and the University/College name. 
       - Format: "Degree Name - University Name" (e.g., "B.Tech Computer Science - Stanford University").
       - Look for keywords like "University", "Institute", "College", "B.Tech", "B.Sc", "M.Tech", "Bachelor", "Master".
       - If the specific degree is not clear, provide the University name.
    5. "role": Identify the SINGLE most likely Job Role this candidate is targeting (e.g., "Data Scientist", "Java Developer", "Digital Marketer").
       - Return ONLY the role name.

    Return strictly valid JSON.
    If a field is not found, return an empty string "".
    """

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", # Upgraded model for better accuracy
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"❌ Extraction Error: {e}")
        return {"name": "", "email": "", "phone": "", "education": "", "role": ""}

# --- 5. VOICE TRANSCRIPTION ---
try:
    @app.post("/transcribe_audio")
    async def transcribe_audio(file: UploadFile = File(...)):
        print(f"🎤 Transcribing audio file: {file.filename}")
        # 1. Save the uploaded file temporarily
        temp_filename = f"temp_{uuid.uuid4()}.wav"
        
        try:
            content = await file.read()
            with open(temp_filename, "wb") as buffer:
                buffer.write(content)
            
            # 2. Send to Groq Whisper
            with open(temp_filename, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(temp_filename, audio_file.read()), # Correct tuple format for Groq
                    model="whisper-large-v3", 
                    response_format="json",
                    language="en"
                )
            
            text = transcription.text
            print(f"📝 Transcript: {text}")
            
            # 3. Cleanup
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
            return {"text": text}
            
        except Exception as e:
            print(f"❌ Audio Error: {e}") 
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return {"error": str(e)}
except RuntimeError as e:
    if "python-multipart" in str(e):
        print("\n❌ CRITICAL ERROR: 'python-multipart' library is missing.")
        print("   To fix, run: pip install python-multipart\n")
    else:
        raise e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_bridge:app", host="127.0.0.1", port=8000, reload=True)
