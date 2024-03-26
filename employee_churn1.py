import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.combine import SMOTETomek

data = pd.read_csv("D:\\Employee_Churn\\employee_churn.csv")
data = data.drop(columns=['Over18', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'MaritalStatus'], axis=1)
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
le = LabelEncoder()
for i in categorical_cols:
    data[i] = le.fit_transform(data[i])

data = data.drop(columns=['DailyRate', 'HourlyRate', 'MonthlyRate', 'EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis=1)
smk = SMOTETomek()
x_res, y_res = smk.fit_resample(data.drop(columns=['Attrition'], axis=1), data['Attrition'])
final_data = pd.concat([x_res, y_res], axis=1)
final_data.drop_duplicates(inplace=True)
final_data.drop(columns=["Education"], axis=1, inplace=True)
X = final_data.drop(columns=["Attrition"], axis=1)
Y = final_data["Attrition"]
sc = StandardScaler()
scaled_data = sc.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(scaled_data, Y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
predicted_probabilities = rf_model.predict_proba(scaled_data)[:, 1]
final_data['Probability_of_Leaving'] = predicted_probabilities
x_train, x_test, y_train, y_test = train_test_split(scaled_data, final_data['Probability_of_Leaving'], test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(mean_squared_error(predictions, y_test))
print(r2_score(predictions,y_test))
pickle.dump({'model': model, 'scaler': sc}, open('empchurn.pkl', 'wb'))
model = pickle.load(open('empchurn.pkl', 'rb'))
