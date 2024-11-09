import streamlit as st
from utils.data_loader import load_data
from utils.regression import perform_regression
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def main():
    st.title("Interactive Multiple Linear Regression")
    st.sidebar.title("Options")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file:
        data = load_data(uploaded_file)
        st.write("### Data Preview")
        st.write(data.head())

        target = st.sidebar.selectbox("Select the target variable", data.columns)
        features = st.sidebar.multiselect("Select the feature variables", data.columns)

        if st.sidebar.button("Run Regression"):
            if target and features and target not in features:
                model, mse, r2, y_test, y_pred = perform_regression(data, target, features)

                st.write("### Regression Results")
                st.write(f"Mean Squared Error: {mse}")
                st.write(f"R-squared: {r2}")

                st.write("### Actual vs Predicted Values")
                result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                st.write(result_df.head())

                st.write("### Residual Plot")
                residuals = y_test - y_pred
                sns.residplot(x=y_pred, y=residuals, lowess=True)
                plt.xlabel("Predicted")
                plt.ylabel("Residuals")
                st.pyplot(plt.gcf())

                st.write("### Enter values for prediction")
                input_data = [st.number_input(f"{feature}", value=float(data[feature].mean())) for feature in features]
                
                if st.button('Predict'):
                    prediction = model.predict([input_data])
                    st.write(f"Predicted {target}: {prediction[0]}")
            else:
                st.error("Please ensure target is not a feature and that both are selected.")
                
if __name__ == "__main__":
    main()
