import streamlit as st
import pandas as pd
import datetime
import numpy as np
import re
import traceback
import matplotlib.pyplot as plt

# Try importing sklearn — if missing, app will still run but model won't load
sklearn_available = True
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
except Exception as e:
    sklearn_available = False
    sklearn_import_error = str(e)

# Must be first Streamlit command
st.set_page_config(page_title="Mental Health Predictor", layout="centered")

# ---------------------------
# Questionnaire definitions
# (UI preserved exactly)
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
# Severity mapping (same as before)
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
# Dataset-driven model: load & train (robust)
# ---------------------------
DATA_FILENAME = "Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv"

@st.cache_resource
def load_and_train_model(path=DATA_FILENAME):
    info = {"status": "ok", "message": "", "trace": None, "feature_cols": None}
    if not sklearn_available:
        info["status"] = "error"
        info["message"] = "scikit-learn not installed in this environment. ML model disabled."
        info["trace"] = sklearn_import_error
        return None, None, info

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        info["status"] = "error"
        info["message"] = f"Dataset file not found at '{path}'. ML model disabled."
        return None, None, info
    except Exception as e:
        info["status"] = "error"
        info["message"] = f"Error reading dataset: {e}. ML model disabled."
        info["trace"] = traceback.format_exc()
        return None, None, info

    try:
        # Heuristic to detect 9 PHQ columns (keeps your original logic)
        cols = df.columns.tolist()

        def find_phq_columns(columns):
            candidates = []
            for col in columns:
                col_clean = col.strip()
                if re.search(r'phq[\s_\-]?\d', col_clean, re.I):
                    candidates.append(col)
                    continue
                if re.search(r'(^|\W)q[\s_\-]?\d(\W|$)', col_clean, re.I):
                    candidates.append(col)
                    continue
                if re.search(r'item[\s_\-]?\d', col_clean, re.I) or re.search(r'question[\s_\-]?\d', col_clean, re.I):
                    candidates.append(col)
                    continue
            candidates = list(dict.fromkeys(candidates))
            if len(candidates) >= 9:
                def col_key(c):
                    m2 = re.search(r'(\d+)(?!.*\d)', c)
                    if m2:
                        return int(m2.group(1))
                    return 999
                candidates = sorted(candidates, key=col_key)
                return candidates[:9]
            return candidates

        phq_cols = find_phq_columns(cols)

        # fallback to first 9 numeric columns if not found (preserve old behavior)
        if len(phq_cols) < 9:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 9:
                phq_cols = numeric_cols[:9]

        if len(phq_cols) != 9:
            info["status"] = "error"
            info["message"] = f"Could not detect 9 PHQ item columns (found {len(phq_cols)}). ML model disabled."
            return None, None, info

        # Ensure columns exist and are numeric
        X = df[phq_cols].copy()
        X = X.apply(pd.to_numeric, errors='coerce')

        # If values outside 0..3, attempt a scale-down if shape allows (preserve original handling)
        max_val = np.nanmax(X.values)
        min_val = np.nanmin(X.values)
        if np.isfinite(max_val) and max_val > 3 and min_val >= 0:
            X = ((X - min_val) / (max_val - min_val)) * 3.0
            X = X.round().clip(0, 3)

        # Target: happiness.score must exist
        if 'happiness.score' not in df.columns:
            info["status"] = "error"
            info["message"] = "Target column 'happiness.score' not found in dataset. ML model disabled."
            return None, None, info

        y = pd.to_numeric(df['happiness.score'], errors='coerce')

        # Keep only fully valid rows
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]

        if len(X) < 20:
            info["status"] = "error"
            info["message"] = f"Not enough valid rows to train model (rows found: {len(X)}). ML model disabled."
            return None, None, info

        # Train a regressor to predict happiness.score
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        info["message"] = f"Trained happiness regressor on {len(X)} rows. Feature columns used: {phq_cols}."
        info["feature_cols"] = phq_cols
        try:
            from sklearn.metrics import mean_squared_error
            y_pred = model.predict(X_test)
            info["test_mse"] = float(mean_squared_error(y_test, y_pred))
        except Exception:
            pass

        return model, phq_cols, info

    except Exception as e:
        info["status"] = "error"
        info["message"] = f"Unexpected error training model: {e}"
        info["trace"] = traceback.format_exc()
        return None, None, info

# Attempt to load/train model (safe call)
try:
    model, feature_cols, model_info = load_and_train_model()
except Exception:
    model = None
    feature_cols = None
    model_info = {"status": "error", "message": "Model initialization crashed unexpectedly.", "trace": traceback.format_exc()}

# ---------------------------
# Page layout functions (UI preserved exactly)
# ---------------------------
def show_home():
    st.title("Mental Health Predictor")
    st.write("Understand your current mental health with a short, friendly check — based on PHQ-9 and GAD-7 questionnaires.")
    # keep same image
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

    # Initialize session state with None to prevent default selection
    st.session_state.phq_answers = st.session_state.get('phq_answers', [None]*len(PHQ9))
    st.session_state.gad_answers = st.session_state.get('gad_answers', [None]*len(GAD7))

    st.subheader("PHQ-9 (Depression)")
    for i, q in enumerate(PHQ9):
        # The index is set to None to avoid a default selection
        st.session_state.phq_answers[i] = st.radio(
            f"{i+1}. {q}",
            options=list(options.keys()),
            format_func=lambda x: options[x],
            index=st.session_state.phq_answers[i],
            key=f"phq_{i}"
        )

    st.subheader("GAD-7 (Anxiety)")
    for i, q in enumerate(GAD7):
        # The index is set to None to avoid a default selection
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
            # Check if all questions were answered
            if all(ans is not None for ans in st.session_state.phq_answers) and all(ans is not None for ans in st.session_state.gad_answers):
                phq9_score = sum(st.session_state.phq_answers)
                gad7_score = sum(st.session_state.gad_answers)

                phq9_severity_label = score_to_severity_phq9(phq9_score)  # default rule-based severity

                # If ML model available, try to use it to predict happiness.score (safe try/except)
                pred_happiness = None
                if model is not None and feature_cols is not None:
                    try:
                        # Build user input DataFrame matching feature_cols
                        user_input = pd.DataFrame([st.session_state.phq_answers], columns=feature_cols)
                        user_input = user_input.apply(pd.to_numeric, errors='coerce')
                        user_input = user_input.fillna(0).round().clip(0, 3)
                        pred_val = model.predict(user_input)[0]
                        pred_happiness = float(round(float(pred_val), 2))
                    except Exception:
                        # don't crash - show small warning and fallback to rule-based
                        st.warning("ML prediction failed; using standard rule-based severity. (Model error logged in sidebar.)")
                        model_info.setdefault("last_predict_trace", traceback.format_exc())

                st.session_state.last_result = {
                    'phq9_score': phq9_score,
                    'gad7_score': gad7_score,
                    'phq9_severity': phq9_severity_label,
                    'gad7_severity': score_to_severity_gad7(gad7_score),
                    'pred_happiness': pred_happiness,
                    'time': datetime.datetime.now().isoformat()
                }
                st.session_state.page = "result"
                st.rerun()
            else:
                st.warning("Please answer all questions before submitting.")

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
    st.write(f"Depression: {res['phq9_severity']} — (PHQ-9 score {res['phq9_score']})")
    st.write(f"Anxiety: {res['gad7_severity']} — (GAD-7 score {res['gad7_score']})")

    st.subheader("Personalized Suggestions")
    for s in suggestions_for(res['phq9_score'], res['gad7_score']):
        st.write(f"- {s}")

    # Show ML prediction if available
    if res.get('pred_happiness') is not None:
        st.subheader("AI-based prediction")
        pred = res['pred_happiness']
        st.write(f"- Predicted happiness score (from dataset): *{pred} / 10*")
        if pred <= 2:
            st.error("Low Happiness (possible depressive symptoms)")
        elif pred <= 6:
            st.warning("Medium Happiness (neutral/moderate mood)")
        else:
            st.success("High Happiness (positive mood)")

    st.subheader("Quick Actions")
    if st.button("Start 5-min breathing exercise"):
        st.info("Box breathing: Inhale 4s — Hold 4s — Exhale 4s — Hold 4s. Repeat for 5 minutes.")

    if st.button("View Resources & Helplines"):
        st.session_state.page = 'resources'

    st.write("---")
    st.subheader("Save your result (optional)")

    # Save result form — clicking Save triggers a feedback panel (expander) if checked
    with st.form("save_result_form", clear_on_submit=False):
        name = st.text_input("Enter a name or nickname (optional)", key="save_name")
        save = st.checkbox("Save my result locally (on this device)", key="save_checkbox")
        save_btn = st.form_submit_button("Save Result")
        # If user clicked Save Result and checkbox checked: prepare pending row and show feedback expander
        if save_btn:
            if save:
                # prepare row but DO NOT write to disk yet — wait for feedback panel
                pending_row = {
                    'name': name or 'anonymous',
                    'phq9_score': res['phq9_score'],
                    'gad7_score': res['gad7_score'],
                    'phq9_severity': res['phq9_severity'],
                    'gad7_severity': res['gad7_severity'],
                    'pred_happiness': res.get('pred_happiness'),
                    'time': res['time']
                }
                st.session_state.pending_row = pending_row
                st.session_state.show_feedback_panel = True
                # Reroute to a new page
                st.session_state.page = "feedback_page"
                st.rerun()
            else:
                st.warning("Please check 'Save my result locally' to save and give feedback.")

    # show graph comparing user's PHQ responses to dataset average
    if feature_cols is not None:
        st.write("---")
        st.subheader("📊 Your PHQ-9 Responses vs Dataset Average")
        try:
            # Check if all questions were answered before attempting to graph
            if all(ans is not None for ans in st.session_state.phq_answers):
                user_responses = [st.session_state.phq_answers[i] for i in range(len(PHQ9))]
                df_full = pd.read_csv(DATA_FILENAME)
                df_full_phq = df_full[feature_cols].apply(pd.to_numeric, errors='coerce')
                avg_values = df_full_phq.mean().values
                indices = np.arange(len(feature_cols))
                fig, ax = plt.subplots(figsize=(8, 4.5))
                bar_width = 0.35
                ax.bar(indices, user_responses, bar_width, label="Your responses", alpha=0.8)
                ax.bar(indices + bar_width, avg_values, bar_width, label="Dataset average", alpha=0.7)
                ax.set_xticks(indices + bar_width / 2)
                ax.set_xticklabels([str(c) for c in feature_cols], rotation=45)
                ax.set_xlabel("PHQ items")
                ax.set_ylabel("Score (0–3)")
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("Please answer all questions to see the comparison chart.")
        except Exception:
            st.info("Could not render comparison chart (dataset may be missing or malformed).")

    if st.button("Take assessment again"):
        st.session_state.page = 'assessment'

    if st.button("Back to Home"):
        st.session_state.page = 'home'

def show_feedback_page():
    st.header("Feedback — optional")
    st.write("Thanks for saving your result! Your feedback helps improve this tool.")
    
    # Check if there is pending data
    row = st.session_state.get('pending_row', None)
    if row is None:
        st.error("Pending data not found — please go back to the results page.")
        if st.button("Back to Results"):
            st.session_state.page = 'result'
            st.rerun()
        return

    st.markdown("### 1) Rate this app (5 stars)")
    star_options = ["★☆☆☆☆", "★★☆☆☆", "★★★☆☆", "★★★★☆", "★★★★★"]
    # Set index to None for no default selection
    star_choice = st.radio("", star_options, index=None, horizontal=True, key="fb_star_choice")
    feedback_stars = star_options.index(star_choice) + 1 if star_choice is not None else None

    st.markdown("### 2) Quick survey (1 = Strongly disagree, 5 = Strongly agree)")
    q_options = [
        "1 - Strongly disagree",
        "2 - Disagree",
        "3 - Neutral",
        "4 - Agree",
        "5 - Strongly agree"
    ]
    # Set index to None for no default selection
    q1 = st.radio("The assessment was easy to understand.", q_options, index=None, horizontal=True, key="fb_q1")
    q2 = st.radio("The predicted result felt relevant to my experience.", q_options, index=None, horizontal=True, key="fb_q2")
    q3 = st.radio("I would trust this app to monitor my mood over time.", q_options, index=None, horizontal=True, key="fb_q3")
    q4 = st.radio("I would recommend this tool to a friend.", q_options, index=None, horizontal=True, key="fb_q4")

    def q_to_int(q_str):
        try:
            return int(q_str.split(" - ")[0])
        except Exception:
            return None

    q1_val = q_to_int(q1)
    q2_val = q_to_int(q2)
    q3_val = q_to_int(q3)
    q4_val = q_to_int(q4)

    st.markdown("### 3) Any comments or suggestions? (optional)")
    comments = st.text_area("", key="fb_comments", height=100)

    contact = st.checkbox("I agree to be contacted for follow-up (optional)", key="fb_contact")

    col_a, col_b = st.columns(2)
    with col_a:
        submit_fb = st.button("Submit Feedback", key="submit_fb")
    with col_b:
        skip_fb = st.button("Skip and Save", key="skip_fb")

    if submit_fb or skip_fb:
        # Check if all survey questions were answered if submitting
        if submit_fb and (star_choice is None or q1 is None or q2 is None or q3 is None or q4 is None):
            st.warning("Please answer all survey questions or click 'Skip and Save'.")
            st.stop()

        # retrieve pending row
        row = st.session_state.pop('pending_row', None)
        if row is None:
            st.error("Pending data not found — nothing to save.")
            st.session_state.page = "result"
            st.rerun()
            return
        
        # Attach feedback if submitted
        if submit_fb:
            row['feedback_stars'] = int(feedback_stars)
            row['feedback_q1'] = int(q1_val) if q1_val is not None else None
            row['feedback_q2'] = int(q2_val) if q2_val is not None else None
            row['feedback_q3'] = int(q3_val) if q3_val is not None else None
            row['feedback_q4'] = int(q4_val) if q4_val is not None else None
            row['feedback_comments'] = comments.strip() if comments else ""
            row['feedback_contact_ok'] = bool(contact)
        else:
            row['feedback_stars'] = None
            row['feedback_q1'] = None
            row['feedback_q2'] = None
            row['feedback_q3'] = None
            row['feedback_q4'] = None
            row['feedback_comments'] = ""
            row['feedback_contact_ok'] = False

        # Save to results.csv (append)
        try:
            df_local = pd.read_csv('results.csv')
            df_local = pd.concat([df_local, pd.DataFrame([row])], ignore_index=True)
        except FileNotFoundError:
            df_local = pd.DataFrame([row])
        df_local.to_csv('results.csv', index=False)

        st.session_state.page = "thank_you_page"
        st.rerun()

    if st.button("Back to Results"):
        st.session_state.page = "result"
        st.rerun()

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
        df_local = pd.read_csv('results.csv')
        df_local['time'] = pd.to_datetime(df_local['time'])
        df_local = df_local.sort_values('time')
        st.line_chart(df_local[['phq9_score','gad7_score']].rename(columns={'phq9_score':'PHQ-9','gad7_score':'GAD-7'}))
        st.dataframe(df_local)
    except FileNotFoundError:
        st.info('No saved results found. Save results from the Results page to track progress.')

    if st.button('Back to Home'):
        st.session_state.page = 'home'

def show_thank_you_page():
    st.header("Thank You!")
    st.success("Your result has been saved and your feedback submitted. Thank you for using our tool and for helping us improve it.")
    st.balloons()
    if st.button("Back to Home"):
        st.session_state.page = 'home'
    if st.button("View your results"):
        st.session_state.page = 'progress'

# ---------------------------
# Navigation (UI preserved)
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

    # Show model status / diagnostic info (non-intrusive)
    if model_info is None:
        st.caption("ML model: not initialized.")
    else:
        if model_info.get("status") == "ok":
            st.caption("ML model: trained from local dataset.")
        else:
            st.caption("ML model inactive — using rule-based severity.")
            # Show short developer-friendly message and allow expanding trace
            with st.expander("Model diagnostics (expand for details)"):
                st.write(model_info.get("message", "No message"))
                trace = model_info.get("trace")
                if trace:
                    st.text_area("Traceback", trace, height=200)
                if model_info.get("message"):
                    st.write(model_info.get("message"))
                if model_info.get("test_mse") is not None:
                    st.write(f"Test MSE: {model_info.get('test_mse'):.3f}")
                if model_info.get("last_predict_trace"):
                    st.text_area("Last prediction traceback", model_info.get("last_predict_trace"), height=200)

# Page rendering (UI preserved)
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
elif page == 'feedback_page':
    show_feedback_page()
elif page == 'thank_you_page':
    show_thank_you_page()
else:
    show_home()
