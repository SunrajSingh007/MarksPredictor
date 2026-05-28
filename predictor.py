import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


df = pd.read_csv('students.csv')

print(df.head())
print(df.shape)
print(df.describe())
print(df.isnull().sum())

print(df.corr(numeric_only=True)['final_marks'])

plt.scatter(df['study_hours'], df['final_marks'], color='purple')
plt.xlabel('Study Hours')
plt.ylabel('Final Marks')
plt.title('Study Hours vs Final Marks')
plt.show()


X = df[['study_hours', 'attendance', 'prev_score']]
y = df['final_marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")


model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")


y_predicted = model.predict(X_test)

print("\nActual vs Predicted:")
for actual, predicted in zip(y_test, y_predicted):
    print(f"  Actual: {actual}  |  Predicted: {predicted:.1f}")


new_student = [[6, 80, 65]]
prediction = model.predict(new_student)
print(f"\nPredicted marks for new student: {prediction[0]:.1f}")


r2 = r2_score(y_test, y_predicted)
mae = mean_absolute_error(y_test, y_predicted)

print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f} marks")

if r2 >= 0.85:
    print("Excellent model!")
elif r2 >= 0.70:
    print("Good model.")
else:
    print("Model needs more data.")


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Student Marks Predictor — Analysis', fontsize=14)

axes[0].scatter(y_test, y_predicted, color='steelblue', alpha=0.7)
axes[0].plot([30, 100], [30, 100], 'r--')
axes[0].set_xlabel('Actual Marks')
axes[0].set_ylabel('Predicted Marks')
axes[0].set_title('Actual vs Predicted')

axes[1].scatter(df['study_hours'], df['final_marks'], color='mediumseagreen', alpha=0.7)
axes[1].set_xlabel('Study Hours')
axes[1].set_ylabel('Final Marks')
axes[1].set_title('Study Hours vs Marks')

features = ['Study Hours', 'Attendance', 'Prev Score']
coefs = model.coef_
axes[2].bar(features, coefs, color=['steelblue', 'coral', 'mediumseagreen'])
axes[2].set_title('Feature Importance')
axes[2].set_ylabel('Coefficient')

plt.tight_layout()
plt.savefig('results.png')
plt.show()
