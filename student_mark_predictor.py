import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
from io import StringIO

# --- Configuration ---
MODEL_PATH = Path(__file__).parent / "marks_predictor_model_v3.joblib"
RANDOM_STATE = 42

# --- Student_Marks.csv content loaded directly ---
# This string contains the content of the CSV you uploaded.
CSV_CONTENT = """number_courses,time_study,Marks
3,4.508,19.202
4,0.096,7.734
4,3.133,13.811
6,7.909,53.018
8,7.811,55.299
6,3.211,17.822
3,6.063,29.889
5,3.413,17.264
4,4.410,20.348
3,6.173,30.862
3,7.353,42.036
7,0.423,12.132
7,4.218,24.318
3,4.274,17.672
3,2.908,11.397
4,4.260,19.466
5,5.719,30.548
8,6.080,38.490
6,7.711,50.986
8,3.977,25.133
4,4.733,22.073
6,6.126,35.939
5,2.051,12.209
7,4.875,28.043
4,3.635,16.517
3,1.407,6.623
7,0.508,12.647
8,4.378,26.532
5,0.156,9.333
4,1.299,8.837
8,3.864,24.172
3,1.923,8.100
8,0.932,15.038
6,6.594,39.965
3,4.083,17.171
3,7.543,43.978
4,2.966,13.119
6,7.283,46.453
7,6.533,41.358
6,7.775,51.142
4,0.140,7.336
6,2.754,15.725
6,3.591,19.771
5,1.557,10.429
4,1.954,9.742
3,2.061,8.924
4,3.797,16.703
4,4.779,22.701
3,5.635,26.882
5,3.913,19.106
6,6.703,40.602
6,4.130,22.184
4,0.771,7.892
7,6.049,36.653
8,7.591,53.158
7,2.913,18.238
8,7.641,53.359
7,7.649,51.583
3,6.198,31.236
8,7.468,51.343
6,0.376,10.522
4,2.438,10.844
6,3.606,19.590
3,4.869,21.379
7,0.130,12.591
6,2.142,13.562
4,5.473,27.569
3,0.550,6.185
4,1.395,8.920
6,3.948,21.400
4,3.736,16.606
5,2.518,13.416
3,4.633,20.398
3,1.629,7.014
4,6.954,39.952
3,0.803,6.217
5,6.379,36.746
8,5.985,38.278
7,7.451,49.544
3,0.805,6.349
7,7.957,54.321
8,2.262,17.705
4,7.410,44.099
5,3.197,16.106
8,1.982,16.461
8,6.201,39.957
7,4.067,23.149
3,1.033,6.053
5,1.803,11.253
7,6.376,40.024
7,4.182,24.394
8,2.730,19.564
4,5.027,23.916
8,6.471,42.426
8,3.919,24.451
6,3.561,19.128
3,0.301,5.609
4,7.163,41.444
7,0.309,12.027
3,6.335,32.357
"""


# ==============================================================================
# 1. DATA LOADING AND MODEL TRAINING
# ==============================================================================

@st.cache_data
def load_and_prepare_data(csv_content_string):
    """Loads and cleans the data from the CSV string."""
    try:
        # Read the CSV content string into a DataFrame
        df = pd.read_csv(StringIO(csv_content_string))

        # Check for required columns
        required_cols = ['number_courses', 'time_study', 'Marks']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Data error: Required columns missing. Found: {list(df.columns)}")
            return None

        # Ensure correct data types and drop NaN rows
        df['number_courses'] = pd.to_numeric(df['number_courses'], errors='coerce')
        df['time_study'] = pd.to_numeric(df['time_study'], errors='coerce')
        df['Marks'] = pd.to_numeric(df['Marks'], errors='coerce')

        df = df.dropna(subset=required_cols)

        if df.empty or len(df) < 5:
            st.error("Training data must contain at least 5 complete, valid rows.")
            return None

        return df

    except Exception as e:
        st.error(f"Error processing CSV data: {e}")
        return None


@st.cache_resource
def train_and_save_model(df):
    """Trains the Random Forest Regressor and saves the pipeline."""

    # Features: Number of courses and time studied
    X = df[['number_courses', 'time_study']]
    # Target: Marks
    y = df['Marks']

    # Define the pipeline: Scaling + Random Forest Regressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE))
    ])

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    pipeline.fit(X_train, y_train)

    # Save model (optional, for persistence across sessions)
    joblib.dump(pipeline, MODEL_PATH)

    # Cross-validation R^2 score for performance indication
    r2_score = pipeline.score(X_test, y_test)

    return pipeline, r2_score


def predict_score(model, input_data, df):
    """Makes a prediction for a single student."""
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)

    # Clip the prediction to the realistic range seen in the training data
    min_mark = df['Marks'].min()
    max_mark = df['Marks'].max()

    return np.clip(prediction[0], min_mark, max_mark)


# ==============================================================================
# 2. STREAMLIT APP LAYOUT
# ==============================================================================

def main():
    st.set_page_config(page_title="Student Mark Predictor", layout="wide")
    st.title("🎓 Student Mark Predictor (Auto-Trained)")
    st.markdown("Model automatically trained on your provided `Student_Marks.csv` data.")

    # --- Load Data and Train Model Automatically ---
    data_df = load_and_prepare_data(CSV_CONTENT)

    if data_df is None:
        st.error("Cannot proceed. Training data failed to load or clean properly.")
        st.stop()

    # Train model (uses st.cache_resource, so it only trains once)
    model, r2_score = train_and_save_model(data_df)

    st.sidebar.markdown("### Model Status")
    st.sidebar.success(f"Trained on {len(data_df)} samples.")
    st.sidebar.info(f"Model R² Score: {r2_score:.2f}")

    st.markdown("---")

    col_input, col_results = st.columns([1, 1])

    # --- INPUT WIDGETS ---
    with col_input:
        st.header("1. Input Student Data for Prediction")

        # Determine realistic input ranges from the data
        min_courses = int(data_df['number_courses'].min())
        max_courses = int(data_df['number_courses'].max())
        min_time = int(data_df['time_study'].min())
        max_time = int(data_df['time_study'].max())

        # Set slider default values to roughly the median/mean of the data
        default_courses = int(np.median(data_df['number_courses']))
        default_time = int(np.median(data_df['time_study']))

        num_courses = st.slider(
            "Number of Courses Opted",
            min_value=min_courses, max_value=max_courses, value=default_courses, step=1,
            help=f"Range based on your data: {min_courses} to {max_courses}."
        )

        time_study = st.slider(
            "Average Time Studied per Day (Hours)",
            min_value=min_time, max_value=max_time, value=default_time, step=1,
            help=f"Range based on your data: {min_time} to {max_time} hours."
        )

    # --- PREDICTION AND RESULTS ---
    with col_results:
        st.header("2. Prediction & Model Status")

        input_data = {
            'number_courses': num_courses,
            'time_study': time_study,
        }

        # Run prediction whenever inputs change
        predicted_mark = predict_score(model, input_data, data_df)

        st.metric(
            label="Predicted Mark (Mark Value)",
            value=f"{predicted_mark:.2f}",
            delta=f"R² Score: {r2_score:.2f}",
            delta_color="off"
        )

        # Dynamic feedback based on max mark in the dataset
        max_df_mark = data_df['Marks'].max()
        if predicted_mark < max_df_mark * 0.3:
            st.warning("Prediction suggests a very low score. High risk.")
        elif predicted_mark > max_df_mark * 0.7:
            st.success("Prediction suggests a strong performance.")
        else:
            st.info("Predicted score is in the mid-range.")

        st.markdown(f"***Training Data Range:*** {data_df['Marks'].min():.2f} to {data_df['Marks'].max():.2f}")

    st.markdown("---")
    st.subheader("3. Historical Training Data Preview")
    st.dataframe(data_df, use_container_width=True)


if __name__ == "__main__":
    main()