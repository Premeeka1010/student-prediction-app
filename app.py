import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, KFold
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Student  Marks & Grade Prediction",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; text-align: center;}
    .sub-header {font-size: 1.5rem; color: #ff7f0e;}
    .prediction-box {background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;}
    .feature-importance {background-color: #e6f7ff; padding: 15px; border-radius: 10px; margin: 10px 0;}
    .grade-A-plus {color: #006400; font-weight: bold; font-size: 1.5rem;}
    .grade-A {color: #228B22; font-weight: bold; font-size: 1.5rem;}
    .grade-B-plus {color: #32CD32; font-weight: bold; font-size: 1.5rem;}
    .grade-B {color: #9ACD32; font-weight: bold; font-size: 1.5rem;}
    .grade-C-plus {color: #FFD700; font-weight: bold; font-size: 1.5rem;}
    .grade-C {color: #FFA500; font-weight: bold; font-size: 1.5rem;}
    .grade-D-plus {color: #FF8C00; font-weight: bold; font-size: 1.5rem;}
    .grade-F {color: #FF4500; font-weight: bold; font-size: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# Function to convert marks to grade
def marks_to_grade(marks):
    if marks >= 90:
        return "A+", "grade-A-plus"
    elif marks >= 80:
        return "A", "grade-A"
    elif marks >= 70:
        return "B+", "grade-B-plus"
    elif marks >= 60:
        return "B", "grade-B"
    elif marks >= 50:
        return "C+", "grade-C-plus"
    elif marks >= 40:
        return "C", "grade-C"
    elif marks >= 30:
        return "D+", "grade-D-plus"
    else:
        return "F", "grade-F"

# Generate sample data (increased to 2000 samples as in the notebook)
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_samples = 2000  # Changed from 200 to 2000
    
    data = {
        'Parental Education': np.random.choice(['High School', 'Bachelor', 'Master', 'Doctorate'], n_samples),
        'Previous Overall Marks': np.random.randint(40, 100, n_samples),
        'Class Attendance (%)': np.round(np.random.uniform(50, 100, n_samples), 2),
        'Extracurricular Activities': np.random.choice(['Yes', 'No'], n_samples),
        'Availability of Resources': np.random.choice(['Yes', 'No'], n_samples),
        'Motivation Level': np.random.randint(2, 6, n_samples),
        'Study Hours per Week': np.random.randint(10, 41, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on the same logic as in the notebook
    # Reduced weights to ensure marks don't exceed 100
    weights = {
        'Previous Overall Marks': 0.3,  # Reduced from 0.5
        'Class Attendance (%)': 0.15,   # Reduced from 0.2
        'Study Hours per Week': 0.1,    # Reduced from 0.15
        'Motivation Level': 2.0,        # Reduced from 3.0
        'Parental Education': 1.0,      # Reduced from 1.50
        'Extracurricular Activities': 0.3,  # Reduced from 0.50
        'Availability of Resources': 1.0    # Reduced from 1.50
    }
    
    edu_map = {
        'High School': 0,
        'Bachelor': 1,
        'Master': 2,
        'Doctorate': 3
    }
    
    future_marks = np.zeros(len(df))
    
    for col, weight in weights.items():
        if col == 'Parental Education':
            future_marks += weight * df[col].map(edu_map).fillna(0) * 5  # Reduced from 20 to 5
        elif col in ['Previous Overall Marks', 'Class Attendance (%)', 'Motivation Level', 'Study Hours per Week']:
            future_marks += weight * df[col]
        else:
            future_marks += weight * pd.factorize(df[col])[0] * 5  # Reduced from 20 to 5
    
    # Add base value and scale to ensure marks are in reasonable range
    future_marks = 20 + (future_marks * 0.8)  # Adjusted scaling
    
    noise = np.random.normal(0, 2.0, len(df))
    future_marks += noise
    future_marks = np.clip(future_marks, 0, 100)
    future_marks = np.round(future_marks, 2)
    
    df['Future Marks'] = future_marks
    return df

# Load or train the model
@st.cache_resource
def load_model(df):
    X = df.drop("Future Marks", axis=1)
    y = df["Future Marks"]
    
    # Define feature types
    num_features = ['Previous Overall Marks', 'Class Attendance (%)', 'Motivation Level', 'Study Hours per Week']
    cat_features = ['Parental Education', 'Extracurricular Activities', 'Availability of Resources']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ], remainder="drop")
    
    model = Pipeline([
        ("pre", preprocessor),
        ("model", LinearRegression())
    ])
    
    # Perform cross-validation
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_validate(model, X, y, cv=cv, 
                           scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'])
    
    mae = -scores['test_neg_mean_absolute_error'].mean()
    rmse = math.sqrt(-scores['test_neg_mean_squared_error'].mean())
    r2 = scores['test_r2'].mean()
    
    # Train the model on the full dataset
    model.fit(X, y)
    
    return model, {'MAE': mae, 'RMSE': rmse, 'R²': r2}

# Main app
def main():
    st.markdown('<h1 class="main-header">🎓 Student Marks & Grade Prediction</h1>', unsafe_allow_html=True)
    st.write("This app uses a trained Linear Regression model to predict student future marks based on various features.")
    
    # Generate data and load model
    df = generate_sample_data()
    model, metrics = load_model(df)
    
    # Display dataset information
    st.sidebar.header("Navigation")
    section = st.sidebar.radio("Go to", ["Data Exploration", "Prediction", "Model Information", "Grade System"])
    
    if section == "Data Exploration":
        st.markdown('<h2 class="sub-header">Student Data Exploration</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Dataset Info")
            st.write(f"Dataset shape: {df.shape}")
            st.write("Columns:", list(df.columns))
            
            if st.checkbox("Show data types"):
                st.write(df.dtypes)
                
            if st.checkbox("Show missing values"):
                st.write(df.isnull().sum())
        
        with col2:
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            if st.checkbox("Show full dataset"):
                st.dataframe(df)
        
        st.subheader("Data Statistics")
        st.write(df.describe())
        
        st.subheader("Feature Distributions")
        selected_feature = st.selectbox("Select feature to visualize", df.columns[:-1])
        
        if df[selected_feature].dtype == 'object':
            value_counts = df[selected_feature].value_counts()
            st.bar_chart(value_counts)
        else:
            fig, ax = plt.subplots()
            sns.histplot(df[selected_feature], kde=True, ax=ax)
            st.pyplot(fig)
    
    elif section == "Prediction":
        st.markdown('<h2 class="sub-header">Predict Future Marks</h2>', unsafe_allow_html=True)
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                parental_education = st.selectbox(
                    "Parental Education",
                    options=['High School', 'Bachelor', 'Master', 'Doctorate']
                )
                
                previous_marks = st.slider(
                    "Previous Overall Marks",
                    min_value=40, max_value=100, value=70
                )
                
                attendance = st.slider(
                    "Class Attendance (%)",
                    min_value=50.0, max_value=100.0, value=75.0, step=0.1
                )
                
                extracurricular = st.selectbox(
                    "Extracurricular Activities",
                    options=['Yes', 'No']
                )
            
            with col2:
                resources = st.selectbox(
                    "Availability of Resources",
                    options=['Yes', 'No']
                )
                
                motivation = st.slider(
                    "Motivation Level",
                    min_value=2, max_value=5, value=3
                )
                
                study_hours = st.slider(
                    "Study Hours per Week",
                    min_value=10, max_value=40, value=30
                )
            
            submitted = st.form_submit_button("Predict Future Marks")
        
        if submitted:
            # Create input data frame
            input_data = pd.DataFrame({
                'Parental Education': [parental_education],
                'Previous Overall Marks': [previous_marks],
                'Class Attendance (%)': [attendance],
                'Extracurricular Activities': [extracurricular],
                'Availability of Resources': [resources],
                'Motivation Level': [motivation],
                'Study Hours per Week': [study_hours]
            })
            
            # Make prediction
            try:
                prediction = model.predict(input_data)
                # Ensure prediction doesn't exceed 100
                prediction = min(prediction[0], 100)
                
                # Convert to grade
                grade, grade_class = marks_to_grade(prediction)
                
                # Display prediction
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.success(f"Predicted Future Marks: **{prediction:.2f}**")
                st.markdown(f'<p class="{grade_class}">Predicted Grade: <strong>{grade}</strong></p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show feature importance for linear regression
                try:
                    # Get feature names after preprocessing
                    num_features = ['Previous Overall Marks', 'Class Attendance (%)', 'Motivation Level', 'Study Hours per Week']
                    cat_features = ['Parental Education', 'Extracurricular Activities', 'Availability of Resources']
                    
                    # Get feature names from the preprocessor
                    cat_transformer = model.named_steps['pre'].named_transformers_['cat']
                    cat_names = cat_transformer.get_feature_names_out(cat_features)
                    feature_names = list(num_features) + list(cat_names)
                    
                    # Get coefficients from linear regression
                    coefficients = model.named_steps['model'].coef_
                    importances = np.abs(coefficients)
                    
                    # Create a DataFrame for visualization
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=True)
                    
                    st.markdown('<div class="feature-importance">', unsafe_allow_html=True)
                    st.subheader("Feature Importances (Absolute Coefficients)")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(importance_df['Feature'], importance_df['Importance'])
                    ax.set_xlabel('Absolute Coefficient Value')
                    ax.set_title('Linear Regression Feature Importances')
                    st.pyplot(fig)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not display feature importances: {str(e)}")
            
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    elif section == "Model Information":
        st.markdown('<h2 class="sub-header">Model Performance</h2>', unsafe_allow_html=True)
        
        st.write("The Linear Regression model was trained on synthetic data with the following performance metrics (from 3-fold cross-validation):")
        
        # Display metrics from cross-validation
        metrics_df = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'R²'],
            'Value': [metrics['MAE'], metrics['RMSE'], metrics['R²']]
        })
        st.table(metrics_df)
        
        st.write("The model was evaluated using 3-fold cross-validation on the generated data.")
        
        st.subheader("Model Details")
        st.write("""
        - **Algorithm**: Linear Regression
        - **Preprocessing**: StandardScaler for numerical features, OneHotEncoder for categorical features
        - **Features used**: 
            - Parental Education
            - Previous Overall Marks
            - Class Attendance (%)
            - Extracurricular Activities
            - Availability of Resources
            - Motivation Level
            - Study Hours per Week
        """)
    
    elif section == "Grade System":
        st.markdown('<h2 class="sub-header">Grade System Information</h2>', unsafe_allow_html=True)
        
        st.write("The following grading system is used to evaluate student performance:")
        
        # Create a table for grade system
        grade_data = {
            'Grade': ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'F'],
            'Marks Range': ['90-100', '80-89.9', '70-79.9', '60-69.9', '50-59.9', '40-49.9', '30-39.9', 'Below 30'],
            'Performance': ['Outstanding', 'Excellent', 'Very Good', 'Good', 'Satisfactory', 'Average', 'Below Average', 'Fail']
        }
        
        grade_df = pd.DataFrame(grade_data)
        st.table(grade_df)
        
        # Add visual examples
        st.subheader("Grade Examples")
        
        # Create example marks and their corresponding grades
        example_marks = [95, 85, 75, 65, 55, 45, 35, 25]
        example_grades = [marks_to_grade(mark)[0] for mark in example_marks]
        example_classes = [marks_to_grade(mark)[1] for mark in example_marks]
        
        # Display examples
        for i, (mark, grade, grade_class) in enumerate(zip(example_marks, example_grades, example_classes)):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"{mark} marks:")
            with col2:
                st.markdown(f'<p class="{grade_class}"><strong>{grade}</strong></p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()