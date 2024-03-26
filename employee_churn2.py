import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

try:
    with open('empchurn.pkl', 'rb') as file:
        model_data = pickle.load(file)
        model = model_data['model']
        scaler = model_data['scaler']
except (IOError, EOFError) as e:
    print("Error Loading the pickled file!!!!", e)

@application.route('/')
def fun():
    return render_template('employee_churn3.html')

@application.route('/prediction', methods=['POST'])
def prediction():
    data = request.form
    features = [[int(data["Age"]), int(data["DistanceFromHome"]),
                 int(data["EnvironmentSatisfaction"]), int(data["JobInvolvement"]), int(data["JobLevel"]),
                 int(data["JobRole"]), int(data["JobSatisfaction"]), int(data["MonthlyIncome"]),
                 int(data["NumCompaniesWorked"]), int(data["OverTime"]), int(data["PercentSalaryHike"]),
                 int(data["PerformanceRating"]), int(data["RelationshipSatisfaction"]),
                 int(data["StockOptionLevel"]), int(data["TotalWorkingYears"]), int(data["TrainingTimesLastYear"]),
                 int(data["WorkLifeBalance"]), int(data["YearsAtCompany"]), int(data["YearsInCurrentRole"]),
                 int(data["YearsSinceLastPromotion"]), int(data["YearsWithCurrManager"])]]

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features) # Convert to percentage
    prediction_percentage=prediction*100
    return render_template("employee_churn3.html", prediction=prediction_percentage)
if __name__ == "__main__":
    application.run(debug=True)
