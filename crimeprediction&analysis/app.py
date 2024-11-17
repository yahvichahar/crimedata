import joblib
import json
import pandas as pd
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

# Load the trained polynomial regression model
model = joblib.load('models/polynomial_regression_model.pkl')

# Load crime data from the JSON file (for years 2020-2023)
def load_crime_data():
    try:
        with open('data/yearly_crime_counts.json', 'r') as file:
            crime_data = json.load(file)
        return crime_data
    except Exception as e:
        print(f"Error loading crime data: {e}")
        return {}

# Load predicted crime data from the JSON file (for years 2024-2029)
def load_predicted_crime_data():
    try:
        with open('data/predicted_crime_counts.json', 'r') as file:
            predicted_data = json.load(file)
        return predicted_data
    except Exception as e:
        print(f"Error loading predicted crime data: {e}")
        return {}

# Load crime data from JSON (for gender-based crime rate distribution)
def load_victim_gender_data():
    try:
        # Load the victim gender crime data from the JSON file
        with open('data/victim_gender_data.json', 'r') as file:
            victim_gender_data = json.load(file)
        return victim_gender_data
    except Exception as e:
        print(f"Error loading victim gender data: {e}")
        return {}

# Load crime data based on severity (Severe, Moderate, Minor)
# Load the dataset
def load_severity_data():
    # Update the file name to crime_classification.csv
    return pd.read_csv('data/crime_classification.csv')


# Route to render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for crime count prediction page
@app.route('/crime_count_prediction', methods=['GET', 'POST'])
def crime_count_prediction():
    prediction = None
    actual_crime_count = None  # For the actual crime count
    error_message = None  # For error feedback
    crime_data = load_crime_data()  # Load the crime data from JSON
    predicted_data = load_predicted_crime_data()  # Load the predicted crime data from JSON

    if request.method == 'POST':
        try:
            # Get the year input from the form
            data = request.form.get('year')

            # Check if the input is empty or not a valid number
            if not data:
                error_message = "Please enter a valid year."
            elif not data.isdigit():
                error_message = "Please enter a valid integer for the year."
            else:
                # Convert input to integer
                year = int(data)

                # Check if the year is between 2020 and 2023 (inclusive)
                if 2020 <= year <= 2023:
                    if str(year) in crime_data:
                        # If year exists in the data, show the actual crime count
                        actual_crime_count = crime_data[str(year)]
                        prediction = None  # Don't show prediction if actual value exists
                    else:
                        error_message = f"Actual crime data for {year} is missing."

                # Check if the year is between 2024 and 2029 (inclusive)
                elif 2024 <= year <= 2029:
                    if str(year) in predicted_data:
                        # Use the predicted data if year exists in the predicted dataset
                        prediction = predicted_data[str(year)]
                        actual_crime_count = None  # Don't show actual count if predicting
                    else:
                        error_message = f"Predicted crime data for {year} is missing."

                else:
                    error_message = "Please enter a year between 2020 and 2029."

        except Exception as e:
            error_message = f"Error: {e}"  # Show any error messages

    return render_template('crime_count_prediction.html', 
                           prediction=prediction, 
                           actual_crime_count=actual_crime_count, 
                           error_message=error_message)

# Route for crime rate by gender page
@app.route('/crime_rate_by_gender', methods=['GET', 'POST'])
def crime_rate_by_gender():
    # Load victim gender-based crime rate data
    victim_gender_data = load_victim_gender_data()

    # Prepare the data for rendering
    try:
        # Convert the loaded data to a dictionary for easier rendering in the template
        victim_gender_dict = victim_gender_data
    except Exception as e:
        victim_gender_dict = {}
        print(f"Error processing crime data: {e}")

    # Handle form data for gender and year filters
    selected_genders = request.form.getlist('gender')
    selected_years = request.form.getlist('year')

    # Filter the data based on selected gender and year
    if selected_genders or selected_years:
        filtered_data = {}
        for year, gender_data in victim_gender_dict.items():
            # Filter by gender
            if selected_genders:
                gender_data = {k: v for k, v in gender_data.items() if k in selected_genders}

            # Filter by year
            if selected_years:
                if str(year) not in selected_years:
                    continue

            if gender_data:  # If there's any data left after filtering
                filtered_data[year] = gender_data
        victim_gender_dict = filtered_data

    return render_template('crime_rate_by_gender.html', victim_gender_dict=victim_gender_dict)

# Route for crime classification page
@app.route('/crime_classification', methods=['GET', 'POST'])
def crime_classification():
    severity_data = load_severity_data()  # Load severity data from CSV

    # Initialize variables for filtered data
    filtered_data = pd.DataFrame()  # Default to empty DataFrame
    selected_severity = []  # Initialize as an empty list to avoid NoneType issues
    num_results = None  # Number of results to display

    if request.method == 'POST':
        # Get selected severities and number of results from the form
        selected_severity = request.form.getlist('severity')  # Get selected severity list
        num_results = request.form.get('num_results')  # Get the number of results

        # Filter the data based on selected severities
        if selected_severity:
            # Filter based on the 'severity_label' column in the dataset
            filtered_data = severity_data[severity_data['severity_label'].isin(selected_severity)]
        else:
            filtered_data = severity_data  # Show all data if no severity is selected

        # Handle the "num_results" input
        if num_results and num_results.isdigit():
            num_results = int(num_results)
            filtered_data = filtered_data.head(num_results)

    # Calculate total records
    total_records = len(severity_data)

    return render_template(
        'crime_classification.html',
        filtered_data=filtered_data,
        total_records=total_records,
        selected_severity=selected_severity,
        num_results=num_results,
    )



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
