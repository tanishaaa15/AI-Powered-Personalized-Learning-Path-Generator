import pandas as pd
import altair as alt
import streamlit as st
from streamlit_option_menu import option_menu
import requests
from fpdf import FPDF
import re
import os
import urllib.parse
import plotly.graph_objects as go
import json
from datetime import datetime
import logging

# Suppress harmless warnings from fontTools regarding 'MERG' tables in Nirmala.ttf
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Career Path Generator", layout="wide")
BACKEND_URL = "http://127.0.0.1:8000"

# --- HISTORY PERSISTENCE ---
# We use the absolute path to ensure the file is found even after reload
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading history: {e}")
            return []
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"❌ Error saving history: {e}")

# --- HELPER FUNCTION: RADAR CHART ---
def create_radar_chart(present_skills, missing_skills):
    # 1. Combine all unique skills to form the axes of the chart
    all_skills = list(set(present_skills + missing_skills))[:10] # Limit to top 10 to keep chart clean
    
    if not all_skills:
        return None

    # 2. Create the "Ideal" Profile (The Job Requirement)
    # We assume the job requires 10/10 in everything.
    required_values = [10] * len(all_skills)
    
    # 3. Create the "User" Profile (Current Status)
    # If skill is present -> 7 (Good). If missing -> 2 (Basic/None).
    user_values = []
    for skill in all_skills:
        if skill in present_skills:
            user_values.append(7)
        else:
            user_values.append(2)
            
    # Close the loop for the chart (repeat first value at the end)
    all_skills.append(all_skills[0])
    required_values.append(required_values[0])
    user_values.append(user_values[0])

    # 4. Build the Plot
    fig = go.Figure()

    # Layer 1: The Job Requirement (Outer Blue Shape)
    fig.add_trace(go.Scatterpolar(
        r=required_values,
        theta=all_skills,
        fill='toself',
        name='Required Mastery',
        line_color='rgba(0, 0, 255, 0.5)',
        fillcolor='rgba(0, 0, 255, 0.1)' 
    ))

    # Layer 2: Your Profile (Inner Orange Shape)
    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=all_skills,
        fill='toself',
        name='Your Level',
        line_color='rgba(255, 165, 0, 0.8)',
        fillcolor='rgba(255, 165, 0, 0.4)'
    ))

    # Styling
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10] # Scale from 0 to 10
            )
        ),
        showlegend=True,
        title="Skill Gap Visualization"
    )
    
    return fig

# --- STATE MANAGEMENT ---
if 'analysis_results' not in st.session_state: st.session_state['analysis_results'] = None
if 'roadmap_ready' not in st.session_state: st.session_state['roadmap_ready'] = False
if 'roadmap_data_from_api' not in st.session_state: st.session_state['roadmap_data_from_api'] = None
if 'resume_text' not in st.session_state: st.session_state['resume_text'] = ""
if "messages" not in st.session_state: st.session_state.messages = []
if "session_id" not in st.session_state: st.session_state.session_id = None

if 'current_resume_name' not in st.session_state: st.session_state['current_resume_name'] = "Unknown Resume"
if 'current_file_id' not in st.session_state: st.session_state['current_file_id'] = ""
if 'detected_role' not in st.session_state: st.session_state['detected_role'] = ""
if 'history' not in st.session_state: st.session_state['history'] = load_history()

# --- RESTORE STATE FROM HISTORY ---
if st.session_state['history']:
    current_resume = st.session_state.get('current_resume_name', "Unknown Resume")

    # 1. Restore Skill Gap Analysis
    if st.session_state['analysis_results'] is None:
        for item in reversed(st.session_state['history']):
            if item['type'] == "Skill Gap Analysis":
                history_resume = item.get('resume_id', "Unknown Resume")
                # Only restore if we are starting fresh OR if history matches current file
                if current_resume == "Unknown Resume" or current_resume == history_resume:
                    st.session_state['analysis_results'] = item['data']
                    if current_resume == "Unknown Resume" and 'resume_id' in item:
                        st.session_state['current_resume_name'] = item['resume_id']
                    break
    
    # Update local var in case it changed during step 1
    current_resume = st.session_state.get('current_resume_name', "Unknown Resume")
    
    # 2. Restore Roadmap
    if st.session_state['roadmap_data_from_api'] is None:
        for item in reversed(st.session_state['history']):
            if item['type'] == "Roadmap Generation" and isinstance(item.get('data'), dict):
                # Check if it contains the full roadmap data
                if 'roadmap' in item['data']:
                    history_resume = item.get('resume_id', "Unknown Resume")
                    if current_resume == "Unknown Resume" or current_resume == history_resume:
                        st.session_state['roadmap_data_from_api'] = item['data']
                        st.session_state['roadmap_ready'] = True
                        break
    
    # 3. Restore Profile
    if 'user_profile' not in st.session_state:
        for item in reversed(st.session_state['history']):
            if item['type'] == "Profile Extraction":
                history_resume = item.get('resume_id', "Unknown Resume")
                if current_resume == "Unknown Resume" or current_resume == history_resume:
                    st.session_state['user_profile'] = item['data']
                    if 'role' in item['data']:
                        st.session_state['detected_role'] = item['data']['role']
                    break

# --- ROBUST PDF GENERATOR (Supports All Regional Languages) ---
@st.cache_resource
def get_font_path():
    return os.path.join(BASE_DIR, "Nirmala.ttf")

def generate_pdf(roadmap_list, target_role, weeks, total_hours, profile=None, interview_score=None):
    pdf = FPDF()
    pdf.add_page()
    
    # 1. SETUP FONT
    font_path = get_font_path()
    
    if os.path.exists(font_path):
        # fpdf2 syntax for adding fonts is cleaner
        pdf.add_font('Nirmala', fname=font_path)
        pdf.set_font('Nirmala', size=12)
    else:
        # Fallback
        pdf.set_font("Helvetica", size=12)

    # 2. HEADER
    pdf.set_font_size(16)
    pdf.cell(0, 10, text="Mastery Learning Roadmap", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.set_font_size(12)
    pdf.cell(0, 10, text=f"Role: {target_role} | {weeks} Weeks", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(5)

    # 2.5 PROFILE & STATUS SECTION (New Task 8)
    if profile:
        pdf.set_font_size(14)
        pdf.cell(0, 10, text="Candidate Profile", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font_size(12)
        pdf.cell(0, 6, text=f"Name: {profile.get('name', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, text=f"Email: {profile.get('email', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

    if interview_score:
        pdf.set_font_size(14)
        pdf.cell(0, 10, text="Interview Status", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font_size(12)
        pdf.cell(0, 6, text=f"Latest Mock Interview Score: {interview_score}/10", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

    # 3. ROADMAP CONTENT LOOP
    for i, week in enumerate(roadmap_list):
        # Week Title
        pdf.set_text_color(0, 0, 139) # Blue
        topic = week.get('week_topic', 'Topic')
        pdf.cell(0, 10, text=f"WEEK {i+1}: {topic}", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_text_color(0, 0, 0) # Black
        
        # Course
        course = week.get('recommended_course', '')
        if course:
            pdf.cell(0, 6, text=f"Official Course: {course}", new_x="LMARGIN", new_y="NEXT")
            
        # Daily Breakdown
        days = week.get('daily_breakdown', [])
        for day in days:
            pdf.set_x(20) # Indent
            day_text = f"- {day.get('day_topic', 'Task')}"
            # Multi_cell handles text wrapping automatically
            pdf.multi_cell(0, 6, text=day_text)
            
        pdf.ln(5)

    # 4. OUTPUT
    # fpdf2 returns bytes directly, so no encoding needed!
    return bytes(pdf.output())

# --- SIDEBAR MENU ---
with st.sidebar:
    selected = option_menu(
        "Main Menu", 
        ["Home", "My Profile", "Skill Gap Analysis", "Learning Roadmap", "Placement Hub", "AI Mock Interview", "Progress Tracker"], 
        icons=['house', 'person-circle', 'bar-chart', 'map', 'briefcase', 'robot', 'graph-up-arrow'], 
        menu_icon="cast", 
        default_index=0,
        key="main_menu_nav_final"
    )

# --- PAGE 1: HOME ---
if selected == "Home":
    st.title("📄 AI Resume Profiler")
    uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])
    
    if uploaded_file is not None:
        # Check if a new file is uploaded to reset previous analysis
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if file_id != st.session_state.get('current_file_id', ''):
            st.session_state['analysis_results'] = None
            st.session_state['roadmap_ready'] = False
            st.session_state['roadmap_data_from_api'] = None
            if 'user_profile' in st.session_state:
                del st.session_state['user_profile']
            st.session_state['detected_role'] = ""
            st.session_state['messages'] = []
            st.session_state['session_id'] = None
            st.session_state['current_file_id'] = file_id

        import pdfplumber
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
        st.session_state['resume_text'] = text
        st.session_state['current_resume_name'] = uploaded_file.name
        st.success("✅ Resume Parsed!")

    if st.session_state['resume_text']:
        st.divider()
        st.subheader("👤 Profile Overview")
        email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", st.session_state['resume_text'])
        email = email_match.group(0) if email_match else "Not detected"
        st.write(f"**Email:** {email}")
        with st.expander("View Raw Text"):
            st.text_area("Content", st.session_state['resume_text'], height=150)
        st.info("Head to **Skill Gap Analysis** to identify your career path.")

# --- PAGE 1.5: MY PROFILE (New) ---
elif selected == "My Profile":
    st.title("👤 My Candidate Profile")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write("Review and edit your auto-extracted details.")
    with col2:
        reload_btn = st.button("🔄 Reload Data")
    
    if 'resume_text' not in st.session_state or not st.session_state['resume_text']:
        st.warning("⚠️ Please upload your resume on the Home page first.")
    else:
        # 1. Check if we already extracted the profile OR if reload requested.
        if 'user_profile' not in st.session_state or reload_btn:
            with st.spinner("🤖 Reading resume to auto-fill details..."):
                try:
                    payload = {"resume_text": st.session_state['resume_text']}
                    res = requests.post(f"{BACKEND_URL}/extract_profile", json=payload)
                    
                    if res.status_code == 200:
                        st.session_state['user_profile'] = res.json()
                        if 'role' in st.session_state['user_profile']:
                            new_role = st.session_state['user_profile']['role']
                            st.session_state['detected_role'] = new_role
                            
                            # Keep Skill Gap Target Role exactly the same as the detected role
                            input_key = f"target_role_{st.session_state.get('current_file_id', 'default')}"
                            st.session_state[input_key] = new_role
                            st.session_state['analysis_results'] = None
                            
                        st.success("Data extracted!")

                        # Log to History
                        item = {
                            "type": "Profile Extraction",
                            "resume_id": st.session_state['current_resume_name'],
                            "data": st.session_state['user_profile'],
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.session_state['history'].append(item)
                        save_history(st.session_state['history'])
                    else:
                        st.error("Failed to extract data.")
                        st.session_state['user_profile'] = {} # Empty fallback
                except requests.exceptions.ConnectionError:
                    st.error("❌ Backend is offline. Please run 'uvicorn api_bridge:app --reload'")
                    st.session_state['user_profile'] = {}
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state['user_profile'] = {}

        # 2. Display the Form for Editing
        profile = st.session_state.get('user_profile', {})
        
        with st.form("profile_form"):
            c1, c2 = st.columns(2)
            with c1:
                new_name = st.text_input("Full Name", value=profile.get('name', ''))
                new_email = st.text_input("Email", value=profile.get('email', ''))
            with c2:
                new_phone = st.text_input("Phone Number", value=profile.get('phone', ''))
                new_edu = st.text_input("Education / Degree", value=profile.get('education', ''))
            
            # Additional Fields (Manual Entry)
            new_linkedin = st.text_input("LinkedIn URL (Optional)", placeholder="https://linkedin.com/in/...")
            new_role = st.text_input("Detected Target Role", value=profile.get('role', st.session_state.get('detected_role', '')))
            
            submitted = st.form_submit_button("💾 Save Profile")
            
            if submitted:
                # Update Session State with Edited Values
                st.session_state['user_profile'] = {
                    "name": new_name,
                    "email": new_email,
                    "phone": new_phone,
                    "education": new_edu,
                    "linkedin": new_linkedin,
                    "role": new_role
                }
                st.session_state['detected_role'] = new_role
                
                # Automatically apply the saved role to the Skill Gap Analysis
                input_key = f"target_role_{st.session_state.get('current_file_id', 'default')}"
                st.session_state[input_key] = new_role
                st.session_state['analysis_results'] = None
                
                st.success("✅ Profile Updated Successfully!")
                st.balloons()

# --- PAGE 2: SKILL GAP ---
elif selected == "Skill Gap Analysis":
    st.title("📊 Skill Gap Analysis")

    if not st.session_state['resume_text'] and not st.session_state['analysis_results']:
        st.warning("Please upload a resume on Home page.")
    else:
        # Use existing results to pre-fill the input if available
        default_role = ""
        if st.session_state.get('detected_role'):
            default_role = st.session_state['detected_role']
        elif 'user_profile' in st.session_state and st.session_state['user_profile'].get('role'):
            default_role = st.session_state['user_profile']['role']
        elif st.session_state['analysis_results'] and st.session_state['analysis_results'].get('role'):
            default_role = st.session_state['analysis_results'].get('role')
        
        input_key = f"target_role_{st.session_state.get('current_file_id', 'default')}"
        
        if input_key not in st.session_state or st.session_state[input_key] == "":
            st.session_state[input_key] = default_role
            
        target_role = st.text_input("Target Role (Optional)", value=default_role, key=input_key,
                                    help="Leave empty to Auto-Detect role from resume (Great for Freshers!).")
        
        if st.button("Analyze Gaps", type="primary"):
            if not st.session_state['resume_text']:
                st.error("Resume text missing. Please upload a resume on Home page to analyze.")
            else:
                with st.spinner("Analyzing against O*NET standards..."):
                    payload = {
                        "resume_text": st.session_state['resume_text'],
                        "target_role": target_role
                    }
                    try:
                        res = requests.post(f"{BACKEND_URL}/analyze_skills", json=payload)
                        if res.status_code == 200:
                            st.session_state['analysis_results'] = res.json()
                            # Log to History
                            item = {
                                "type": "Skill Gap Analysis",
                                "resume_id": st.session_state['current_resume_name'],
                                "data": st.session_state['analysis_results'],
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.session_state['history'].append(item)
                            save_history(st.session_state['history'])
                            st.rerun()
                        else:
                            st.error("Backend Error.")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Backend is offline. Please start the server.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Display Results
        if st.session_state['analysis_results']:
            res = st.session_state['analysis_results']
            
            st.divider()
            # Show the specific role (Auto-detected or User-defined)
            st.subheader(f"Match Score: {res.get('match_percentage', 0)}% for '{res.get('role')}'")
            st.progress(res.get('match_percentage', 0) / 100)
            
            # --- RADAR CHART VISUALIZATION ---
            present = res.get('present_skills', [])
            missing = res.get('missing_skills', [])
            
            st.markdown("### 🕸️ Skill Radar")
            
            # Call the helper function defined at the top of app.py
            radar_fig = create_radar_chart(present, missing)
            
            if radar_fig:
                st.plotly_chart(radar_fig)
            else:
                st.warning("Not enough skills detected to generate a radar chart.")
            
            # --- SKILL LISTS ---
            c1, c2 = st.columns(2)
            with c1: 
                st.success(f"✅ Verified ({len(present)})")
                for s in present: st.write(f"• {s}")
            with c2:
                st.error(f"❌ Gaps ({len(missing)})")
                for s in missing: st.write(f"• {s}")

# --- REPLACE THE 'Learning Roadmap' SECTION IN app.py ---

elif selected == "Learning Roadmap":
    st.title("🗺️ Personalized Learning Roadmap")
    
    if not st.session_state['analysis_results']:
        st.warning("Run Skill Gap Analysis first.")
    else:
        role = st.session_state['analysis_results'].get('role', 'Professional')
        
        # --- UPDATE THIS INPUT SECTION IN app.py ---
        
        c1, c2, c3 = st.columns(3)
        with c1: weeks = st.slider("Duration (Weeks)", 2, 12, 4)
        with c2: hours = st.slider("Study Hours/Day", 1, 6, 2)
        with c3: 
            # EXPANDED LANGUAGE LIST (All supported by Nirmala UI)
            lang = st.selectbox(
                "Language", 
                [
                    "English", 
                    "Hindi", 
                    "Marathi", 
                    "Bengali", 
                    "Telugu", 
                    "Tamil", 
                    "Gujarati", 
                    "Kannada", 
                    "Malayalam", 
                    "Odia", 
                    "Punjabi", 
                    "Assamese"
                ]
            )
        
        if st.button("Generate Roadmap", type="primary"):
            if not st.session_state['resume_text']:
                st.error("Resume text missing. Please upload a resume on Home page.")
            else:
                with st.spinner("Fetching NCVET Standards & Generating Plan..."):
                    payload = {
                        "resume_text": st.session_state['resume_text'],
                        "target_role": role,
                        "weeks": weeks,
                        "daily_hours": hours,
                        "language": lang
                    }
                    try:
                        res = requests.post(f"{BACKEND_URL}/generate_roadmap", json=payload)
                        if res.status_code == 200:
                            data = res.json()
                            st.session_state['roadmap_data_from_api'] = data
                            st.session_state['roadmap_ready'] = True
                            # Log to History
                            item = {
                                "type": "Roadmap Generation",
                                "resume_id": st.session_state['current_resume_name'],
                                "data": data,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.session_state['history'].append(item)
                            save_history(st.session_state['history'])
                            st.rerun()
                        else:
                            st.error(f"Server Error: {res.status_code}")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Backend is offline. Please start the server.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state['roadmap_ready'] and st.session_state['roadmap_data_from_api']:
            data = st.session_state['roadmap_data_from_api']
            roadmap = data.get('roadmap', [])
            
            if not roadmap:
                st.warning("⚠️ Empty roadmap returned. Please try again.")
            else:
                st.subheader("💾 Export")
                
                # --- CRASH-PROOF PDF BLOCK ---
                try:
                    # Gather data for the report
                    user_profile = st.session_state.get('user_profile', {})
                    # Find latest interview score
                    last_score = "N/A"
                    for item in reversed(st.session_state['history']):
                        if item['type'] == "Interview Result":
                            last_score = item.get('score', "N/A")
                            break
                            
                    pdf = generate_pdf(roadmap, role, weeks, weeks*hours*5, user_profile, last_score)
                    st.download_button("Download Career Report (PDF)", data=pdf, file_name="Career_Report.pdf", mime="application/pdf")
                except Exception as e:
                    st.warning(f"⚠️ PDF Export unavailable for {lang} (Font missing). View roadmap below instead.")
                
                st.divider()
                tabs = st.tabs([f"Week {i+1}" for i in range(len(roadmap))])
                for i, week in enumerate(roadmap):
                    with tabs[i]:
                        st.subheader(week.get('week_topic', 'Topic'))
                        
                        course = week.get('recommended_course') or week.get('Recommended_Course') or "General Study"
                        st.info(f"🎓 **Official Course:** {course}")
                        
                        for day in week.get('daily_breakdown', []):
                            with st.expander(f"📅 {day.get('day_topic', 'Day Task')}"):
                                st.write(day.get('description', ''))
                                
                                c1, c2 = st.columns([1, 1])
                                with c1:
                                    st.markdown(f"[📖 Read Docs]({day.get('documentation', '#')})")
                                with c2:
                                    # --- SMART SEARCH FIX ---
                                    raw_query = day.get('video_search', '')
                                    # If query is short (lazy AI), we build a better one
                                    if len(raw_query) < 10:
                                        query = f"{day.get('day_topic')} {week.get('week_topic')} {lang} tutorial"
                                    else:
                                        query = raw_query
                                    
                                    safe_query = urllib.parse.quote(query)
                                    st.markdown(f"[📺 Watch Video](https://www.youtube.com/results?search_query={safe_query})")

# --- PAGE 4: PLACEMENT HUB ---
elif selected == "Placement Hub":
    st.title("💼 Placement Resource Hub")
    st.write("One-stop destination for placement preparation, cheat sheets, and company-specific insights.")
    
    st.divider()

    # --- SECTION 1: DOWNLOADABLE CHEAT SHEETS ---
    st.subheader("📚 Quick Revision Cheat Sheets")
    st.caption("Download concise guides for last-minute revision.")
    
    c1, c2, c3 = st.columns(3)
    
    # Helper function to read file (prevents crash if file missing)
    def read_file(filename):
        try:
            with open(filename, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    with c1:
        st.markdown("#### 🐍 Python")
        st.write("Syntax, Data Structures, OOPs.")
        file_data = read_file("python_cheat_sheet.pdf") # Make sure this file exists in folder
        if file_data:
            st.download_button("Download PDF", data=file_data, file_name="Python_Cheat_Sheet.pdf", mime="application/pdf")
        else:
            st.button("Download PDF", disabled=True, help="File not uploaded to server yet.")

    with c2:
        st.markdown("#### 🗄️ SQL")
        st.write("Joins, Normalization, Queries.")
        file_data = read_file("sql_cheat_sheet.pdf")
        if file_data:
            st.download_button("Download PDF", data=file_data, file_name="SQL_Cheat_Sheet.pdf", mime="application/pdf")
        else:
            st.button("Download PDF", disabled=True, help="File not uploaded to server yet.")

    with c3:
        st.markdown("#### 🧩 Data Structures")
        st.write("Arrays, Trees, Graphs Algorithms.")
        file_data = read_file("dsa_cheat_sheet.pdf")
        if file_data:
            st.download_button("Download PDF", data=file_data, file_name="DSA_Cheat_Sheet.pdf", mime="application/pdf")
        else:
            st.button("Download PDF", disabled=True, help="File not uploaded to server yet.")

    st.divider()

    # --- SECTION 2: COMPANY SPECIFIC PREP ---
    st.subheader("🎯 Target Company Prep")
    company = st.selectbox("Select Target Company", ["TCS", "Infosys", "Wipro", "Accenture", "Google"])
    
    # Mock Data for Companies (You can expand this later)
    company_data = {
        "TCS": {
            "focus": "Aptitude & Basic Coding",
            "topics": ["Numerical Ability", "Verbal Reasoning", "C/Java Basics", "Data Structures"],
            "pattern": "NQT (National Qualifier Test) Pattern: 2 Coding Qs + 40 Aptitude Qs."
        },
        "Infosys": {
            "focus": "Puzzle Solving & Pseudocode",
            "topics": ["Pseudocode", "Puzzle Solving", "Critical Thinking", "Database Basics"],
            "pattern": "InfyTQ Pattern: Hackathon style + MCQ on DBMS/OOPs."
        },
        "Wipro": {
            "focus": "Logical Reasoning & Communication",
            "topics": ["Essay Writing", "Logical Reasoning", "Basic Programming", "Operating Systems"],
            "pattern": "NLTH Pattern: Coding (Easy) + Essay Writing Section."
        },
        "Accenture": {
            "focus": "Cognitive & Technical",
            "topics": ["Pseudocode", "Networking", "Cloud Basics", "Cognitive Assessment"],
            "pattern": "Z-Pattern: Cognitive -> Technical -> Coding."
        },
        "Google": {
            "focus": "Advanced DSA & System Design",
            "topics": ["Dynamic Programming", "Graphs", "System Design", "Concurrency"],
            "pattern": "Standard Pattern: 3-4 Rounds of Hard LeetCode style problems."
        }
    }

    if company:
        data = company_data.get(company, {})
        st.info(f"💡 **Strategy for {company}:** {data.get('pattern')}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 🔥 High Priority Topics")
            for topic in data.get("topics", []):
                st.markdown(f"- {topic}")
        
        with col_b:
            st.markdown("### 📝 Practice Resources")
            st.link_button(f"Practice for {company} on GeeksforGeeks", f"https://www.geeksforgeeks.org/tag/{company.lower()}/")
            st.link_button(f"Practice {company} Interview Questions", f"https://www.interviewbit.com/{company.lower()}-interview-questions/")
                                    
# --- PAGE 4: INTERVIEW ---
elif selected == "AI Mock Interview":
    st.title("🤖 AI Mock Interviewer")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # "audio_key" is the secret to resetting the widget
    if "audio_key" not in st.session_state:
        st.session_state.audio_key = 0

    if not st.session_state.get('resume_text'):
        st.warning("Please upload your resume on the Home page first.")
    else:
        # Start Button
        if st.button("Start Interview", type="primary"):
            st.session_state.messages = []
            
            # Safe Role Detection
            role = "Professional"
            if st.session_state.get('analysis_results'):
                role = st.session_state['analysis_results'].get('role', 'Professional')
            
            try:
                with st.spinner("Connecting to AI Interviewer..."):
                    res = requests.post(f"{BACKEND_URL}/start_interview", json={
                        "resume_text": st.session_state['resume_text'], 
                        "target_role": role
                    }).json()
                    
                    st.session_state.session_id = res['session_id']
                    
                    # Handles both nested and flat responses just in case
                    q_text = res.get('first_question', {}).get('question') or res.get('question')
                    
                    st.session_state.messages.append({"role": "assistant", "content": q_text})
                    st.rerun()
            except requests.exceptions.ConnectionError:
                st.error("❌ Backend is offline. Please start the server.")
            except Exception as e:
                st.error(f"Error: {e}")

        # Chat History
        for msg in st.session_state.messages:
            st.chat_message(msg['role']).write(msg['content'])

        # Check if Interview is Complete
        interview_complete = False
        if st.session_state.messages:
            last_msg = st.session_state.messages[-1]
            if last_msg['role'] == 'assistant' and "Interview Complete" in last_msg['content']:
                interview_complete = True

        if interview_complete:
            st.success("Interview Completed!")
            
            # --- TASK 7: PERSISTENCE (Save Score) ---
            # Extract score from the last message
            last_msg_content = st.session_state.messages[-1]['content']
            match = re.search(r"Score:\s*(\d+)/10", last_msg_content)
            if match:
                score = int(match.group(1))
                # Save to history
                item = {
                    "type": "Interview Result",
                    "resume_id": st.session_state['current_resume_name'],
                    "score": score,
                    "role": st.session_state.get('analysis_results', {}).get('role', 'General'),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                # Avoid duplicates if user refreshes
                if not st.session_state['history'] or st.session_state['history'][-1] != item:
                    st.session_state['history'].append(item)
                    save_history(st.session_state['history'])
                    st.info(f"📈 Score {score}/10 saved to Progress Tracker!")
            
            # Restart Button
            if st.button("🔄 Restart Interview"):
                # Save Transcript to History before clearing
                item = {
                    "type": "Mock Interview Session",
                    "resume_id": st.session_state['current_resume_name'],
                    "data": st.session_state.messages,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state['history'].append(item)
                save_history(st.session_state['history'])
                
                # Reset State
                st.session_state.messages = []
                st.session_state.session_id = None
                st.rerun()
        else:
            # Input Area (Only show if interview is NOT complete)
            input_container = st.container()
            final_answer = None

            with input_container:
                # Dynamic Key for resetting audio widget
                current_key = f"audio_{st.session_state.audio_key}"
                
                audio_value = st.audio_input("🎙️ Record your answer", key=current_key)
                text_input = st.chat_input("Or type your answer here...")

            # CASE A: Handle Voice
            if audio_value:
                with st.spinner("🎧 Transcribing..."):
                    try:
                        files = {"file": ("answer.wav", audio_value, "audio/wav")}
                        try:
                            res = requests.post(f"{BACKEND_URL}/transcribe_audio", files=files)
                        except requests.exceptions.ConnectionError:
                            st.error("❌ Backend not reachable. Is api_bridge.py running?")
                            res = None
                        
                        if res and res.status_code == 200:
                            data = res.json()
                            if "error" in data:
                                st.error(f"Transcription Error: {data['error']}")
                            else:
                                transcript = data.get("text", "")
                                
                                if transcript.strip():
                                    final_answer = transcript
                                    st.success(f"✅ Transcribed: \"{transcript}\" - Processing Answer...")
                                    st.session_state.audio_key += 1
                                else:
                                    st.warning("⚠️ Audio recorded but no speech detected. Try speaking closer to the mic.")
                        elif res and res.status_code == 404:
                            st.error("❌ Endpoint /transcribe_audio not found. Please restart the backend server.")
                        elif res:
                            st.error(f"Backend Error: {res.status_code}")
                    except Exception as e:
                        st.error(f"Audio connection error: {e}")

            # CASE B: Handle Text
            if text_input:
                final_answer = text_input

            if final_answer:
                st.session_state.messages.append({"role": "user", "content": final_answer})
                st.chat_message("user").write(final_answer)
                
                try:
                    with st.spinner("Thinking..."):
                        response = requests.post(f"{BACKEND_URL}/answer", json={
                            "session_id": st.session_state.get('session_id'), 
                            "answer": final_answer
                        })
                        
                        if response.status_code == 200:
                            res = response.json()
                            reply = res.get('next_question', "Error fetching response.")
                            st.session_state.messages.append({"role": "assistant", "content": reply})
                            st.chat_message("assistant").write(reply)
                            st.rerun()
                        else:
                            st.error(f"Server Error {response.status_code}: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Connection Failed. Ensure 'api_bridge.py' is running at {BACKEND_URL}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# --- PAGE 4.5: PROGRESS TRACKER (Task 7) ---
elif selected == "Progress Tracker":
    st.title("📈 Progress Tracker")
    st.write("Track your interview performance over time.")
    
    # Filter history for interview results
    scores = []
    dates = []
    
    for item in st.session_state['history']:
        if item.get('type') == "Interview Result":
            scores.append(item.get('score', 0))
            # Parse date for cleaner chart
            dt = item.get('timestamp', '').split(' ')[0]
            dates.append(dt)
            
    if not scores:
        st.info("No interview scores recorded yet. Complete a Mock Interview to see your progress!")
    else:
        # Create DataFrame for Chart
        df = pd.DataFrame({
            'Date': dates,
            'Score': scores
        })
        
        st.subheader("Interview Score Improvement")
        st.line_chart(df.set_index('Date'))
        
        avg_score = sum(scores) / len(scores)
        st.metric("Average Score", f"{avg_score:.1f} / 10")
