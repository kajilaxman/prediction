import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Create the dataset
data = pd.DataFrame({
    'Study Hours': [1, 2, 3, 4, 5, 6],
    'Marks': [50, 60, 65, 70, 80, 90]
})

# Step 2: Train the Linear Regression Model
X = data[['Study Hours']].values  # Features
y = data['Marks'].values  # Target

model = LinearRegression()
model.fit(X, y)

# Sidebar menu
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "Predicted Marks"])

# Home Page
if selection == "Home":
    st.title("Statistics Overview")

    # Display the dataset
    st.write("### Dataset Used")
    st.write(data)

    # Show basic statistics
    st.write("### Basic Statistics")
    st.write(data.describe())

    # Visualizations
    st.write("### Visualizations")

    # Bar chart of study hours vs marks
    st.write("#### Bar Chart: Study Hours vs Marks")
    st.bar_chart(data.set_index('Study Hours'))

    # Line chart of study hours vs marks
    st.write("#### Line Graph: Study Hours vs Marks")
    fig, ax = plt.subplots()
    ax.plot(data['Study Hours'], data['Marks'], marker='o', linestyle='-', color='b')
    ax.set_title("Line Graph: Study Hours vs Marks")
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Marks")
    st.pyplot(fig)

    # Pie chart of marks distribution
    st.write("#### Pie Chart: Marks Distribution")
    fig, ax = plt.subplots()
    ax.pie(data['Marks'], labels=data['Study Hours'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.set_title("Pie Chart: Marks Distribution by Study Hours")
    st.pyplot(fig)

# Predicted Marks Page
elif selection == "Predicted Marks":
    st.title("Predict Marks Based on Study Hours")

    # User input for hours
    hours = st.number_input("Enter the number of study hours:", min_value=0.0, max_value=24.0, step=0.5)

    # Predict the marks
    if st.button("Predict"):
        new_hours = np.array([[hours]])
        predicted_marks = model.predict(new_hours)
        st.write(f"Predicted Marks for {hours} hours of study: {predicted_marks[0]:.2f}")

    # Display the model parameters
    st.write("### Model Parameters")
    st.write(f"Intercept: {model.intercept_}")
    st.write(f"Coefficient: {model.coef_[0]}")
 




