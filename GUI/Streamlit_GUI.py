import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide", initial_sidebar_state="expanded")
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'prediction_proba' not in st.session_state:
    st.session_state.prediction_proba = None
if 'form_data' not in st.session_state:
    st.session_state.form_data = None
if 'processed_input' not in st.session_state:
    st.session_state.processed_input = None
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Input Data"
if 'last_selected_model' not in st.session_state:
    st.session_state.last_selected_model = None

st.markdown("""
    <style>
    .main {
        background-color: #FAFFFA;
    }
    .stButton>button {
        background-color: #008080;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #006666;
    }
    .stSelectbox, .stNumberInput {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 5px;
        margin: 5px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #ff8c00;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #008080;
    }
    h1, h2, h3 {
        color: #008080;
    }
    .stMarkdown {
        color: #333333;
    }
    .stApp > header {
        height: 40px !important;
        padding: 5px 10px !important;
    }
    .css-1aumxhk {
        min-height: 0px !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("About")
st.sidebar.markdown("""
This app predicts employee attrition using machine learning models.
Select a model and input employee details to get a prediction.
""")

image_path = "assets/Employees_asset.jpg"
if os.path.exists(image_path):
    st.sidebar.image(image_path, caption="Employee Attrition Predictor", use_container_width=True)
else:
    st.sidebar.info("Employee Attrition Predictor")
    st.sidebar.caption("(Image not found: Employees_asset.jpg)")

st.title("Employee Attrition Predictor")
st.markdown("**Predict whether an employee is likely to leave with our advanced ML models.**")

@st.cache_data
def load_data():
    return pd.read_csv('Data/aug_train.csv')

try:
    train_data = load_data()
except FileNotFoundError:
    st.error("Training data file 'aug_train.csv' not found. Please ensure the file is in the same directory.")
    st.stop()

def preprocess_input(data, le_dict, categorical_cols, model_name):
    data = data.copy()
    
    data['gender'] = data['gender'].fillna('Unknown')
    data['enrolled_university'] = data['enrolled_university'].fillna('no_enrollment')
    data['education_level'] = data.apply(
        lambda row: 'Graduate' if row['enrolled_university'] in ['Full time course', 'Part time course']
        else 'High School' if pd.isna(row['education_level']) else row['education_level'], axis=1
    )
    data['major_discipline'] = data['major_discipline'].fillna('No Major')
    data['experience'] = data['experience'].fillna('>20')
    data['company_size'] = data['company_size'].fillna('50-99')
    data['company_type'] = data['company_type'].fillna('Pvt Ltd')
    data['last_new_job'] = data['last_new_job'].fillna('1')
    
    exp_map = {
        '<1': 0.5, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
        '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19,
        '20': 20, '>20': 21
    }
    data['exp_numeric'] = data['experience'].map(exp_map)
    
    data['cdi_experience'] = data['city_development_index'] * data['exp_numeric']
    
    for col in categorical_cols:
        if col in le_dict:
            try:
                data[col] = le_dict[col].transform(data[col])
            except ValueError as e:
                st.error(f"Error encoding {col}: {str(e)}")
                data[col] = 0
    
    features = [
        'city_development_index',
        'company_size',
        'cdi_experience',
        'experience',
        'enrolled_university',
        'relevent_experience',
        'company_type',
        'gender',
        'major_discipline',
        'education_level',
        'last_new_job',
        'training_hours'
        ]    
    return data[features]

@st.cache_resource
def load_label_encoders():
    categorical_cols = ['city', 'gender', 'relevent_experience', 'enrolled_university',
                        'education_level', 'major_discipline', 'experience',
                        'company_size', 'company_type', 'last_new_job']
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(train_data[col].fillna('Unknown'))
        le_dict[col] = le
    return le_dict, categorical_cols

le_dict, categorical_cols = load_label_encoders()

model_options = {
    'LightGBM': 'pkl_models/lightgbm_model.pkl',
    'XGBoost': 'pkl_models/xgboost_model.pkl',
    'Logistic Regression': 'pkl_models/logistic_model.pkl',
    'Neural Network': 'pkl_models/mlp_model.pkl'
}

selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))

if st.session_state.last_selected_model != selected_model:
    st.session_state.prediction = None
    st.session_state.prediction_proba = None
    st.session_state.form_data = None
    st.session_state.processed_input = None
    st.session_state.selected_tab = "Input Data"
    st.session_state.last_selected_model = selected_model

@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found. Please ensure all model files are in the same directory.")
        return None

model = load_model(model_options[selected_model])
if model is None:
    st.stop()

tab1, tab2, tab3 = st.tabs(["Input Data", "Prediction Results", "Model Insights"])

with tab1:
    st.header("Enter Employee Details")
    with st.form(key='employee_form'):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            city = st.selectbox("City", sorted(train_data['city'].unique()))
            gender = st.selectbox("Gender", ['Male', 'Female', 'Other', 'Unknown'])
            relevent_experience = st.selectbox("Relevant Experience",
                                               ['Has relevent experience', 'No relevent experience'])
            enrolled_university = st.selectbox("Enrolled University",
                                               ['no_enrollment', 'Full time course', 'Part time course'])
        
        with col2:
            education_level = st.selectbox("Education Level",
                                           ['Graduate', 'Masters', 'High School', 'Phd', 'Primary School'])
            major_discipline = st.selectbox("Major Discipline",
                                            ['STEM', 'Humanities', 'Other', 'Business Degree', 'Arts', 'No Major'])
            experience = st.selectbox("Experience",
                                      ['<1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                       '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '>20'])
        
        with col3:
            company_size = st.selectbox("Company Size",
                                        ['<10', '10/49', '50-99', '100-500', '500-999', '1000-4999',
                                         '5000-9999', '10000+'])
            company_type = st.selectbox("Company Type",
                                        ['Pvt Ltd', 'Funded Startup', 'Public Sector', 'Early Stage Startup',
                                         'NGO', 'Other'])
            last_new_job = st.selectbox("Last New Job", ['never', '1', '2', '3', '4', '>4'])
            training_hours = st.number_input("Training Hours", min_value=0, max_value=336, value=50)
            city_development_index = st.number_input("City Development Index", min_value=0.0, max_value=1.0, value=0.8)
        
        submit_button = st.form_submit_button(label="Predict Attrition")
    
    if submit_button:
        st.session_state.form_data = {
            'city': city,
            'city_development_index': city_development_index,
            'gender': gender,
            'relevent_experience': relevent_experience,
            'enrolled_university': enrolled_university,
            'education_level': education_level,
            'major_discipline': major_discipline,
            'experience': experience,
            'company_size': company_size,
            'company_type': company_type,
            'last_new_job': last_new_job,
            'training_hours': training_hours
        }
        
        input_data = pd.DataFrame({k: [v] for k, v in st.session_state.form_data.items()})
        
        processed_input = preprocess_input(input_data, le_dict, categorical_cols, selected_model)
        st.session_state.processed_input = processed_input
        
        try:
            prediction = model.predict(processed_input)
            prediction_proba = model.predict_proba(processed_input)[0]
            
            st.session_state.prediction = prediction[0]
            st.session_state.prediction_proba = prediction_proba
            
            st.session_state.selected_tab = "Prediction Results"
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.stop()
        
        result = "Leave" if prediction[0] == 1 else "Stay"
        
        st.markdown("---")
        st.subheader("Quick Prediction Summary")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if prediction[0] == 1:
                st.error(f"**Prediction:** Employee is likely to {result}")
                st.markdown(f"**Risk Level:** High")
            else:
                st.success(f"**Prediction:** Employee is likely to {result}")
                st.markdown(f"**Risk Level:** Low")
            
            st.write(f"**Stay Probability:** {prediction_proba[0]:.2%}")
            st.write(f"**Leave Probability:** {prediction_proba[1]:.2%}")
            st.write(f"**Model Used:** {selected_model}")
        
        with col2:
            st.markdown("**For detailed results and analysis, navigate to the 'Prediction Results' tab above.**")

with tab2:
    st.header("Prediction Results")
    
    if st.session_state.prediction is not None:
        st.write("Processing detailed analysis...")
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
        
        progress.empty()
        
        if st.session_state.prediction == 1:
            st.error("**The employee is likely to leave** (Attrition: Yes)")
            risk_color = "#ff4444"
            risk_text = "HIGH RISK"
        else:
            st.success("**The employee is not likely to leave** (Attrition: No)")
            risk_color = "#00aa00"
            risk_text = "LOW RISK"
        
        st.markdown(f"""
        <div style="
            background-color: {risk_color}20;
            border-left: 5px solid {risk_color};
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        ">
            <h3 style="color: {risk_color}; margin: 0;">Risk Assessment: {risk_text}</h3>
            <p style="margin: 5px 0 0 0;">Based on the provided employee data and {selected_model} model analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Prediction Confidence")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            prediction_proba = st.session_state.prediction_proba
            colors = ['#008080', '#ff8c00']
            bars = ax.barh(['Stay', 'Leave'], [prediction_proba[0], prediction_proba[1]], color=colors)
            
            for i, (bar, prob) in enumerate(zip(bars, [prediction_proba[0], prediction_proba[1]])):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{prob:.1%}', ha='left', va='center', fontweight='bold')
            
            ax.set_xlabel("Probability", fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_title(f"Attrition Prediction Confidence - {selected_model}", fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            if st.session_state.prediction == 1:
                bars[1].set_edgecolor('red')
                bars[1].set_linewidth(3)
            else:
                bars[0].set_edgecolor('green')
                bars[0].set_linewidth(3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Key Statistics")
            confidence = max(prediction_proba[0], prediction_proba[1])
            st.metric("Confidence Level", f"{confidence:.1%}")
            st.metric("Model Accuracy", "77.8%" if selected_model == "XGBoost" else "75.8%")
            
            st.markdown("### Recommendation")
            if st.session_state.prediction == 1:
                st.markdown("""
                - **Immediate Action Required**
                - Consider retention strategies
                - Schedule 1-on-1 meeting
                - Review compensation & benefits
                - Assess job satisfaction
                """)
            else:
                st.markdown("""
                - **Employee likely to stay**
                - Continue regular check-ins
                - Maintain engagement levels
                - Consider for growth opportunities
                """)
        
        st.subheader("Employee Profile Summary")
        if st.session_state.form_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                **Personal Info:**
                - Experience: {st.session_state.form_data['experience']} years
                - Education: {st.session_state.form_data['education_level']}
                - Major: {st.session_state.form_data['major_discipline']}
                - Gender: {st.session_state.form_data['gender']}
                """)
            
            with col2:
                st.markdown(f"""
                **Work History:**
                - Relevant Experience: {st.session_state.form_data['relevent_experience']}
                - Last New Job: {st.session_state.form_data['last_new_job']} year(s) ago
                - Training Hours: {st.session_state.form_data['training_hours']}
                """)
            
            with col3:
                st.markdown(f"""
                **Company Info:**
                - Company Size: {st.session_state.form_data['company_size']}
                - Company Type: {st.session_state.form_data['company_type']}
                - City Development Index: {st.session_state.form_data['city_development_index']:.2f}
                """)
    
    else:
        st.info("Please make a prediction in the **Input Data** tab first to see detailed results here.")
        st.markdown("---")
        st.markdown("""
        **What you'll see here after making a prediction:**
        - Detailed probability charts
        - Risk assessment
        - Actionable recommendations
        - Employee profile analysis
        """)

with tab3:
    st.header("Model Insights")
    
    model_metrics = {
        'LightGBM': {'Accuracy': 0.7576, 'ROC AUC': 0.8017, 'Precision': 0.7234, 'Recall': 0.6891},
        'XGBoost': {'Accuracy': 0.7779, 'ROC AUC': 0.7873, 'Precision': 0.7445, 'Recall': 0.7012},
        'Logistic Regression': {'Accuracy': 0.7500, 'ROC AUC': 0.7600, 'Precision': 0.7100, 'Recall': 0.6800},
        'Neural Network': {'Accuracy': 0.7800, 'ROC AUC': 0.7900, 'Precision': 0.7350, 'Recall': 0.7150}
    }
    
    st.subheader(f"Performance Metrics for {selected_model}")
    
    col1, col2, col3, col4 = st.columns(4)
    metrics = model_metrics[selected_model]
    
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
    with col2:
        st.metric("ROC AUC", f"{metrics['ROC AUC']:.3f}")
    with col3:
        st.metric("Precision", f"{metrics['Precision']:.1%}")
    with col4:
        st.metric("Recall", f"{metrics['Recall']:.1%}")
    
    if st.session_state.processed_input is not None:
        if selected_model in ['LightGBM', 'XGBoost'] and hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            
            feature_importance = pd.DataFrame({
                'Feature': st.session_state.processed_input.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'], 
                          color='lightblue', edgecolor='navy', alpha=0.7)
            
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontsize=9)
            
            ax.set_xlabel("Importance Score", fontsize=12)
            ax.set_title(f"{selected_model} - Feature Importance Analysis", fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("### Key Insights:")
            top_features = feature_importance.head(3)['Feature'].tolist()
            st.markdown(f"""
            - **Most Important Factor:** {top_features[0]}
            - **Second Most Important:** {top_features[1]} 
            - **Third Most Important:** {top_features[2]}
            
            These features have the strongest influence on the attrition prediction.
            """)
            
        elif selected_model == 'Logistic Regression' and hasattr(model, 'coef_'):
            st.subheader("Feature Coefficients")
            
            if st.session_state.processed_input is not None:
                coefficients = pd.DataFrame({
                    'Feature': st.session_state.processed_input.columns,
                    'Coefficient': model.coef_[0]
                }).sort_values(by='Coefficient', key=abs, ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['red' if x < 0 else 'green' for x in coefficients['Coefficient']]
                bars = ax.barh(coefficients['Feature'], coefficients['Coefficient'], 
                              color=colors, alpha=0.7)
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + (0.01 if width > 0 else -0.01), 
                           bar.get_y() + bar.get_height()/2, 
                           f'{width:.3f}', ha='left' if width > 0 else 'right', 
                           va='center', fontsize=9)
                
                ax.set_xlabel("Coefficient Value", fontsize=12)
                ax.set_title(f"{selected_model} - Feature Coefficients", fontsize=14, fontweight='bold')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                ### Coefficient Interpretation:
                - **Positive coefficients** (green) increase the likelihood of attrition
                - **Negative coefficients** (red) decrease the likelihood of attrition
                - **Larger absolute values** indicate stronger influence
                """)
    
    else:
        st.info("Make a prediction first to see feature importance analysis for your specific case.")
    
    st.subheader("Model Comparison")
    
    comparison_df = pd.DataFrame(model_metrics).T
    comparison_df = comparison_df.round(4)
    
    st.dataframe(
        comparison_df.style.highlight_max(axis=0, color='lightblue').format("{:.1%}"),
        use_container_width=True
    )
    
    st.markdown("""
    ### Model Selection Guide:
    - **XGBoost**: Best overall accuracy, good for general predictions
    - **LightGBM**: Highest ROC AUC, best for ranking/probability estimates  
    - **Neural Network**: Good balance of all metrics
    - **Logistic Regression**: Most interpretable, fastest training
    """)
    
    st.subheader("About Employee Attrition")
    st.markdown("""
    **Common factors that influence employee attrition:**
    
    **Education & Experience**: Higher education and relevant experience often correlate with job mobility
    
    **Company Factors**: Company size, type, and growth stage affect retention rates
    
    **Location**: City development index reflects job market opportunities
    
    **Career Development**: Training hours and career progression opportunities impact retention
    
    **Job History**: Frequency of job changes is a strong predictor of future moves
    """)