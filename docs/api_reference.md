# API Reference

## Functions

### load_data(file)
- **Description**: Loads CSV data into a DataFrame.
- **Parameters**: `file` (str): Path to the CSV file.
- **Returns**: DataFrame with loaded data.

### perform_regression(data, target, features)
- **Description**: Runs a linear regression on selected features and target.
- **Parameters**:
  - `data` (DataFrame): Input data.
  - `target` (str): Target variable.
  - `features` (list of str): Feature variables.
- **Returns**: Model, MSE, R-squared, y_test, y_pred.

