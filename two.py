import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

df = pd.read_csv('songs_dataset.csv')

X = df[['danceability']]
y = df['popularity']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error:", root_mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

X_plot = np.linspace(X.min().values[0], X.max().values[0], 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', alpha=0.6, label='Actual Songs')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Polynomial Fit')
plt.title("Song Popularity Prediction using Polynomial Regression")
plt.xlabel("Danceability")
plt.ylabel("Popularity")
plt.legend()
plt.tight_layout()
plt.show()
