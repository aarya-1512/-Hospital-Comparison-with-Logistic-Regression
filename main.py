import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def main():
    try:
        # File to read
        filename = "Week14Assignment.txt"

        # Load the data
        df = pd.read_csv(filename, delimiter=",")
        # Trim spaces from column names
        df.columns = df.columns.str.strip()

        # Uncheck this line to check for column names in the print statement
        # print(f"Columns in the dataset: {df.columns}")

        # Verify required columns
        required_columns = [
            "PatientID", "Readmission", "StaffSatisfaction",
            "CleanlinessSatisfaction", "FoodSatisfaction",
            "ComfortSatisfaction", "CommunicationSatisfaction"
        ]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Calculate statistics
        num_readmitted = df["Readmission"].sum()
        avg_scores = df[["StaffSatisfaction", "CleanlinessSatisfaction", "FoodSatisfaction",
                         "ComfortSatisfaction", "CommunicationSatisfaction"]].mean()

        # Display statistics
        print(f"Number of Patients Readmitted: {num_readmitted}")
        for category, score in avg_scores.items():
            print(f"Average {category}: {score:.2f}")

        # Calculate overall satisfaction
        df["OverallSatisfaction"] = df[[
            "StaffSatisfaction", "CleanlinessSatisfaction", "FoodSatisfaction",
            "ComfortSatisfaction", "CommunicationSatisfaction"
        ]].mean(axis=1)

        # Prepare data for logistic regression
        X = df["OverallSatisfaction"].values.reshape(-1, 1)
        y = df["Readmission"].values

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform logistic regression
        model = LogisticRegression()
        model.fit(X_scaled, y)

        # Logistic regression results
        print("\nLogistic Regression Results:")
        print("-" * 28)
        print(f"Coefficient: {model.coef_[0][0]:.4f}")
        print(f"Intercept: {model.intercept_[0]:.4f}")
        print(f"Model Score (Accuracy): {model.score(X_scaled, y):.4f}")

        # Predict probabilities for visualization
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_plot_scaled = scaler.transform(X_plot)
        y_prob = model.predict_proba(X_plot_scaled)[:, 1]

        # Plot data and logistic regression curve
        plt.scatter(df["OverallSatisfaction"], y, color="blue", label="Data Points")
        plt.plot(X_plot, y_prob, color="red", label="Logistic Regression Curve")
        plt.xlabel("Overall Satisfaction Score")
        plt.ylabel("Probability of Readmission")
        plt.title("Logistic Regression: Satisfaction vs. Readmission")
        plt.legend()
        plt.show()

    except FileNotFoundError:
        print("Error: The file does not exist. Please check the file name and location.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if _name_ == "_main_":
    main()
