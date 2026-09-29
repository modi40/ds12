import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['Pclass'], bins=[0.5, 1.5, 2.5, 3.5], rwidth=0.7, color='steelblue', edgecolor='black')
plt.xticks([1, 2, 3], ['Class 1', 'Class 2', 'Class 3'])
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.subplot(1, 2, 2)
survival_counts = df['Survived'].value_counts()
plt.pie(survival_counts, labels=['Did Not Survive', 'Survived'], autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'], startangle=140)
plt.title("Survival Distribution")

plt.tight_layout()
plt.show()
