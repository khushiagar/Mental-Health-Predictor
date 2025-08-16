import streamlit as st
import pandas as pd
import datetime

# ---------------------------
# Simple Streamlit Mental Health Predictor
# Single-file app using PHQ-9 & GAD-7 scoring rules.
# Saves results optionally to a local CSV (results.csv).
# ---------------------------

st.set_page_config(page_title="Mental Health Predictor", layout="centered")

# ---------------------------
# Questionnaire definitions
# ---------------------------
PHQ9 = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving a lot more than usual",
    "Thoughts that you would be better off dead, or of hurting yourself in some way"
]

GAD7 = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen"
]

options = {
    0: "Not at all",
    1: "Several days",
    2: "More than half the days",
    3: "Nearly every day"
}

# ---------------------------
# Helper functions
# ---------------------------
def score_to_severity_phq9(score):
    if score <= 4:
        return "Minimal or none"
    elif score <= 9:
        return "Mild"
    elif score <= 14:
        return "Moderate"
    elif score <= 19:
        return "Moderately severe"
    else:
        return "Severe"

def score_to_severity_gad7(score):
    if score <= 4:
        return "Minimal or none"
    elif score <= 9:
        return "Mild"
    elif score <= 14:
        return "Moderate"
    else:
        return "Severe"

def suggestions_for(phq9_score, gad7_score):
    suggestions = []
    if phq9_score >= 20 or gad7_score >= 15:
        suggestions.append("High risk — contact a mental health professional or crisis helpline immediately.")
    if phq9_score >= 10 or gad7_score >= 10:
        suggestions.append("Consider scheduling an appointment with a counselor or psychologist.")
    if phq9_score >= 5 or gad7_score >= 5:
        suggestions.append("Daily self-care: 20–30 min walk, regular sleep, limit caffeine, short breathing exercises.")
    if phq9_score <= 4 and gad7_score <= 4:
        suggestions.append("You're doing okay — maintain healthy habits: sleep, exercise, social connections.")

    suggestions.append("Quick exercises you can try now: 1) 5-min guided breathing, 2) 10-min walk, 3) write 3 things you're grateful for.")
    return suggestions

# ---------------------------
# Page layout functions
# ---------------------------
def show_home():
    st.title("Mental Health Predictor")
    st.write("Understand your current mental health with a short, friendly check — based on PHQ-9 and GAD-7 questionnaires.")
    st.image("https://images.unsplash.com/photo-1511497584788-876760111969?auto=format&fit=crop&w=1050&q=80")

    st.markdown("\n*Features:*")
    st.markdown("- Quick assessment (PHQ-9 + GAD-7)\n- Friendly results & suggestions\n- Optional save to track progress\n- Resources & helplines")

    if st.button("Start Assessment"):
        st.session_state.page = "assessment"

    with st.expander("Why this is not a medical diagnosis"):
        st.write("This tool is for awareness only. It does not replace a medical diagnosis. If you're in crisis, contact local emergency services or a helpline.")

def show_assessment():
    st.header("Assessment")
    st.write("Answer the questions based on the last 2 weeks.")

    st.session_state.phq_answers = st.session_state.get('phq_answers', [0]*len(PHQ9))
    st.session_state.gad_answers = st.session_state.get('gad_answers', [0]*len(GAD7))

    st.subheader("PHQ-9 (Depression)")
    for i, q in enumerate(PHQ9):
        st.session_state.phq_answers[i] = st.radio(
            f"{i+1}. {q}",
            options=list(options.keys()),
            format_func=lambda x: options[x],
            index=st.session_state.phq_answers[i],
            key=f"phq_{i}"
        )

    st.subheader("GAD-7 (Anxiety)")
    for i, q in enumerate(GAD7):
        st.session_state.gad_answers[i] = st.radio(
            f"{i+1}. {q}",
            options=list(options.keys()),
            format_func=lambda x: options[x],
            index=st.session_state.gad_answers[i],
            key=f"gad_{i}"
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Home"):
            st.session_state.page = "home"
    with col2:
        if st.button("Submit"):
            phq9_score = sum(st.session_state.phq_answers)
            gad7_score = sum(st.session_state.gad_answers)
            st.session_state.last_result = {
                'phq9_score': phq9_score,
                'gad7_score': gad7_score,
                'phq9_severity': score_to_severity_phq9(phq9_score),
                'gad7_severity': score_to_severity_gad7(gad7_score),
                'time': datetime.datetime.now().isoformat()
            }
            st.session_state.page = "result"

def show_result():
    res = st.session_state.get('last_result', None)
    if not res:
        st.info("No assessment found. Start a new assessment.")
        if st.button("Start Assessment"):
            st.session_state.page = 'assessment'
        return

    st.header("Your Results")
    st.metric(label="PHQ-9 Score", value=res['phq9_score'], delta=f"{res['phq9_severity']}")
    st.metric(label="GAD-7 Score", value=res['gad7_score'], delta=f"{res['gad7_severity']}")

    st.subheader("Interpretation")
    st.write(f"*Depression:* {res['phq9_severity']} — (PHQ-9 score {res['phq9_score']})")
    st.write(f"*Anxiety:* {res['gad7_severity']} — (GAD-7 score {res['gad7_score']})")

    st.subheader("Personalized Suggestions")
    for s in suggestions_for(res['phq9_score'], res['gad7_score']):
        st.write(f"- {s}")

    st.subheader("Quick Actions")
    if st.button("Start 5-min breathing exercise"):
        st.info("Box breathing: Inhale 4s — Hold 4s — Exhale 4s — Hold 4s. Repeat for 5 minutes.")

    if st.button("View Resources & Helplines"):
        st.session_state.page = 'resources'

    st.write("---")
    st.subheader("Save your result (optional)")
    name = st.text_input("Enter a name or nickname (optional)")
    save = st.checkbox("Save my result locally (on this device)")
    if save and st.button("Save Result"):
        row = {
            'name': name or 'anonymous',
            'phq9_score': res['phq9_score'],
            'gad7_score': res['gad7_score'],
            'phq9_severity': res['phq9_severity'],
            'gad7_severity': res['gad7_severity'],
            'time': res['time']
        }
        try:
            df = pd.read_csv('results.csv')
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)  # ✅ fixed append
        except FileNotFoundError:
            df = pd.DataFrame([row])
        df.to_csv('results.csv', index=False)
        st.success('Result saved locally to results.csv')

    if st.button("Take assessment again"):
        st.session_state.page = 'assessment'

    if st.button("Back to Home"):
        st.session_state.page = 'home'

def show_resources():
    st.header("Resources & Helplines")
    st.write("If you are in immediate danger or thinking of harming yourself, contact local emergency services immediately.")
    st.write("\n*India (examples):*\n- Kiran: 1800-599-0019\n- Snehi / Local NGOs: search city-specific services.")

    st.subheader("Recommended Quick Resources")
    st.markdown(
        "- [5-minute guided breathing audio (YouTube)](https://www.youtube.com/results?search_query=5+minute+breathing+exercise)\n"
        "- [Short guided meditation](https://www.youtube.com/results?search_query=short+guided+meditation)\n"
        "- Articles: WHO, local mental health NGOs."
    )

    st.write("---")
    if st.button("Back to Results"):
        st.session_state.page = 'result'
    if st.button("Back to Home"):
        st.session_state.page = 'home'

def show_progress_tracker():
    st.header("Progress Tracker")
    st.write("This reads data from results.csv (if you saved results) and shows a simple trend.")
    try:
        df = pd.read_csv('results.csv')
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        st.line_chart(df[['phq9_score','gad7_score']].rename(columns={'phq9_score':'PHQ-9','gad7_score':'GAD-7'}))
        st.dataframe(df)
    except FileNotFoundError:
        st.info('No saved results found. Save results from the Results page to track progress.')

    if st.button('Back to Home'):
        st.session_state.page = 'home'

# ---------------------------
# Navigation
# ---------------------------
if 'page' not in st.session_state:
    st.session_state.page = 'home'

page = st.session_state.page

with st.sidebar:
    st.title('Menu')
    if st.button('Home'):
        st.session_state.page = 'home'
    if st.button('Take Assessment'):
        st.session_state.page = 'assessment'
    if st.button('Resources & Helplines'):
        st.session_state.page = 'resources'
    if st.button('Progress Tracker'):
        st.session_state.page = 'progress'
    st.write('---')
    st.write('Privacy: This tool does not store data unless you save it locally.')

# Page rendering
if page == 'home':
    show_home()
elif page == 'assessment':
    show_assessment()
elif page == 'result':
    show_result()
elif page == 'resources':
    show_resources()
elif page == 'progress':
    show_progress_tracker()
else:
    show_home()
