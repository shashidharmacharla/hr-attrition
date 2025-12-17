import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Page config
st.set_page_config(page_title="Advanced HR Analytics Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("hr_data_enhanced.csv", parse_dates=["HireDate", "LeaveDate"])
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Employees")

departments = st.sidebar.multiselect("Select Departments", options=df["Department"].unique(), default=df["Department"].unique())
genders = st.sidebar.multiselect("Select Gender", options=df["Gender"].unique(), default=df["Gender"].unique())
attrition_types = st.sidebar.multiselect("Select Attrition Type", options=df["AttritionType"].dropna().unique(), default=df["AttritionType"].dropna().unique())
job_roles = st.sidebar.multiselect("Select Job Role", options=df["JobRole"].unique(), default=df["JobRole"].unique())

filtered_df = df[
    (df["Department"].isin(departments)) &
    (df["Gender"].isin(genders)) &
    (df["JobRole"].isin(job_roles))
]

# Handle AttritionType filter properly (include no attrition if None selected)
if attrition_types:
    filtered_df = filtered_df[(filtered_df["AttritionType"].isin(attrition_types)) | (filtered_df["Attrition"].eq("No"))]
else:
    filtered_df = filtered_df[filtered_df["Attrition"] == "No"]

# KPIs
total_employees = len(filtered_df)
attrition_count = filtered_df["Attrition"].value_counts().get("Yes", 0)
attrition_rate = (attrition_count / total_employees) * 100 if total_employees else 0
avg_salary = filtered_df["MonthlyIncome"].mean()
avg_years_at_company = filtered_df["YearsAtCompany"].mean()

st.title("🧑‍💼 Advanced HR Analytics Dashboard")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Employees", total_employees)
col2.metric("Attrition Count", attrition_count)
col3.metric("Attrition Rate (%)", f"{attrition_rate:.2f}")
col4.metric("Avg Monthly Salary", f"₹{avg_salary:,.0f}")
col5.metric("Avg Years at Company", f"{avg_years_at_company:.1f}")

st.divider()

# Attrition breakdown by Type
st.subheader("Attrition Breakdown")
attrition_breakdown = filtered_df[filtered_df["Attrition"] == "Yes"]["AttritionType"].value_counts()
fig1 = px.pie(names=attrition_breakdown.index, values=attrition_breakdown.values, title="Attrition Type Distribution")
st.plotly_chart(fig1, use_container_width=True)

# Attrition by Department
st.subheader("Attrition Count by Department")
attrition_dept = filtered_df[filtered_df["Attrition"] == "Yes"]["Department"].value_counts()
fig2 = px.bar(attrition_dept, labels={'index':'Department', 'value':'Attrition Count'})
st.plotly_chart(fig2, use_container_width=True)

# Attrition Trend over time (by LeaveDate)
st.subheader("Attrition Trend Over Time")
attrition_time = filtered_df[filtered_df["Attrition"] == "Yes"].copy()
attrition_time['YearMonth'] = attrition_time["LeaveDate"].dt.to_period("M")
attrition_time_count = attrition_time.groupby("YearMonth").size().reset_index(name='Count')
attrition_time_count['YearMonth'] = attrition_time_count['YearMonth'].dt.to_timestamp()
fig3 = px.line(attrition_time_count, x="YearMonth", y="Count", title="Monthly Attrition Count")
st.plotly_chart(fig3, use_container_width=True)

# Salary Distribution
st.subheader("Salary Distribution")
fig4 = px.histogram(filtered_df, x="MonthlyIncome", nbins=20, title="Salary Distribution")
st.plotly_chart(fig4, use_container_width=True)

# Age Distribution
st.subheader("Age Distribution")
fig5 = px.histogram(filtered_df, x="Age", nbins=15, title="Age Distribution")
st.plotly_chart(fig5, use_container_width=True)

# Job Role vs Attrition Heatmap
st.subheader("Job Role Attrition Heatmap")
heatmap_data = pd.crosstab(filtered_df["JobRole"], filtered_df["Attrition"])
fig6 = px.imshow(heatmap_data, text_auto=True, aspect="auto", title="Job Role vs Attrition")
st.plotly_chart(fig6, use_container_width=True)

# Performance Rating Distribution
st.subheader("Performance Rating Distribution")
fig7 = px.histogram(filtered_df, x="PerformanceRating", title="Performance Ratings")
st.plotly_chart(fig7, use_container_width=True)

# Years at Company Distribution
st.subheader("Years at Company Distribution")
fig8 = px.histogram(filtered_df, x="YearsAtCompany", nbins=15, title="Years at Company")
st.plotly_chart(fig8, use_container_width=True)

# Show Data Table
st.subheader("Employee Data Table")
st.dataframe(filtered_df)

st.divider()

# Simple Attrition Prediction ML Model
st.header("🔮 Simple Attrition Prediction Model")

# Prepare dataset for prediction
ml_df = df.dropna(subset=["Attrition"])  # Drop rows with no attrition info
ml_df["AttritionFlag"] = ml_df["Attrition"].apply(lambda x: 1 if x=="Yes" else 0)

# Features and target
features = ["Age", "MonthlyIncome", "YearsAtCompany", "PerformanceRating", "EducationLevel", "YearsInCurrentRole"]
X = ml_df[features]
y = ml_df["AttritionFlag"]

# Convert categorical EducationLevel if needed (if categorical)
if X["EducationLevel"].dtype == 'object':
    X["EducationLevel"] = pd.Categorical(X["EducationLevel"]).codes

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest Classifier
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Show classification report
st.text("Model Performance on Test Data:")
st.text(classification_report(y_test, y_pred))

# Input fields for prediction
st.subheader("Predict Attrition for New Employee")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    income = st.number_input("Monthly Income", min_value=10000, max_value=200000, value=50000)
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=3)
    performance = st.slider("Performance Rating", 1, 5, 3)
    education = st.selectbox("Education Level (0=Below Bachelor,1=Bachelor,2=Master,3=Doctorate)", [0,1,2,3], index=1)
    years_in_role = st.number_input("Years in Current Role", min_value=0, max_value=40, value=1)
    
    submit = st.form_submit_button("Predict Attrition")

if submit:
    input_data = pd.DataFrame({
        "Age": [age],
        "MonthlyIncome": [income],
        "YearsAtCompany": [years_at_company],
        "PerformanceRating": [performance],
        "EducationLevel": [education],
        "YearsInCurrentRole": [years_in_role]
    })
    prediction = model.predict(input_data)[0]
    result = "Likely to Leave (Attrition)" if prediction == 1 else "Likely to Stay"
    st.success(f"Prediction: {result}")

# Download Filtered Data Button
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name='filtered_hr_data.csv',
    mime='text/csv',
)
