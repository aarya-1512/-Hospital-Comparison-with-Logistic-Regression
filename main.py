import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_and_validate_data(filename):
    """Load and validate the dataset."""
    try:
        df = pd.read_csv(filename, delimiter=",")
        df.columns = df.columns.str.strip()
        required_columns = [
            "PatientID", "Readmission", "StaffSatisfaction",
            "CleanlinessSatisfaction", "FoodSatisfaction",
            "ComfortSatisfaction", "CommunicationSatisfaction"
        ]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{filename}' does not exist.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the file: {e}")

def calculate_statistics(df):
    """Calculate readmission count and average satisfaction scores."""
    num_readmitted = df["Readmission"].sum()
    avg_scores = df[[
        "StaffSatisfaction", "CleanlinessSatisfaction", "FoodSatisfaction",
        "ComfortSatisfaction", "CommunicationSatisfaction"
    ]].mean()
    return num_readmitted, avg_scores

def perform_logistic_regression(df):
    """Perform logistic regression to determine correlation between satisfaction and readmission."""
    df["OverallSatisfaction"] = df[[
        "StaffSatisfaction", "CleanlinessSatisfaction", "FoodSatisfaction",
        "ComfortSatisfaction", "CommunicationSatisfaction"
    ]].mean(axis=1)
    X = df["OverallSatisfaction"].values.reshape(-1, 1)
    y = df["Readmission"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    return model, scaler

def plot_logistic_regression(df, model, scaler, hospital_name):
    """Plot logistic regression results."""
    X_plot = np.linspace(df["OverallSatisfaction"].min(), df["OverallSatisfaction"].max(), 100).reshape(-1, 1)
    X_plot_scaled = scaler.transform(X_plot)
    y_prob = model.predict_proba(X_plot_scaled)[:, 1]

    plt.scatter(df["OverallSatisfaction"], df["Readmission"], color="blue", label="Data Points")
    plt.plot(X_plot, y_prob, color="red", label="Logistic Regression Curve")
    plt.xlabel("Overall Satisfaction Score")
    plt.ylabel("Probability of Readmission")
    plt.title(f"{hospital_name}: Satisfaction vs. Readmission")
    plt.legend()

def main():
    try:
        # Load data for both hospitals
        hospital1_file = "Hospital1.txt"
        hospital2_file = "Hospital2.txt"

        df_hospital1 = load_and_validate_data(hospital1_file)
        df_hospital2 = load_and_validate_data(hospital2_file)

        # Calculate statistics for both hospitals
        stats_hospital1 = calculate_statistics(df_hospital1)
        stats_hospital2 = calculate_statistics(df_hospital2)

        # Perform logistic regression for both hospitals
        model1, scaler1 = perform_logistic_regression(df_hospital1)
        model2, scaler2 = perform_logistic_regression(df_hospital2)

        # Display statistics
        print("Hospital Comparison:\n")
        print("Hospital 1 Data Analysis:")
        print("--------------------------")
        print(f"Number of Patients Readmitted: {stats_hospital1[0]}")
        for category, score in stats_hospital1[1].items():
            print(f"Average {category}: {score:.2f}")

        print("\nHospital 2 Data Analysis:")
        print("--------------------------")
        print(f"Number of Patients Readmitted: {stats_hospital2[0]}")
        for category, score in stats_hospital2[1].items():
            print(f"Average {category}: {score:.2f}")

        # Logistic regression results
        print("\nLogistic Regression Results:")
        print("-----------------------------")
        print(f"Hospital 1 Coefficient: {model1.coef_[0][0]:.4f}, Intercept: {model1.intercept_[0]:.4f}")
        print(f"Hospital 2 Coefficient: {model2.coef_[0][0]:.4f}, Intercept: {model2.intercept_[0]:.4f}")

        # Plot results for both hospitals
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plot_logistic_regression(df_hospital1, model1, scaler1, "Hospital 1")

        plt.subplot(1, 2, 2)
        plot_logistic_regression(df_hospital2, model2, scaler2, "Hospital 2")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
