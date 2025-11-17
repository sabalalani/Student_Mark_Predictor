🎓 Student Mark Predictor

This is a Streamlit web application designed to predict a student's mark based on historical academic data. It trains a Random Forest Regression model using the provided Student_Marks.csv data (included directly in the app script).

Features

Auto-Training: The model is automatically loaded and trained using the embedded data upon application launch.

Predictive Sliders: Users can adjust two key features—Number of Courses and Average Time Studied—to see the resulting predicted mark in real-time.

Model Performance: Displays the $R^2$ score of the trained model for quick performance assessment.

Setup and Installation

1. Prerequisites

Ensure you have Python (3.7+) installed.

2. Install Dependencies

Install all necessary libraries using the provided requirements.txt file:

pip install -r requirements.txt


3. Run the Application

Execute the Streamlit script from your terminal:

streamlit run student_mark_predictor.py


The application will automatically open in your default web browser.

Data Schema

The embedded training data (Student_Marks.csv) uses the following schema:

Column Name

Description

Example

number_courses

time_study

Marks
