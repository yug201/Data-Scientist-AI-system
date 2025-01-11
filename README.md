

## Data scientist ai system

---

# AI System for Dataset Analysis and ML Model Creation

## Overview

This AI system is designed to assist with dataset analysis and Machine Learning (ML) model creation. The system takes in a dataset's path and description, along with the task specified by the user (e.g., predicting a specific column). It generates strategies, executes them step-by-step, adjusts the strategy based on analysis and outputs, and handles errors effectively. All actions, including code execution, output, analysis, and strategy adjustments, are documented in a structured markdown file.

### Key Features:
- **Data Analysis**: The system automatically analyzes datasets, providing useful insights and recommendations.
- **ML Model Creation**: It generates and executes code for creating ML models based on user requests.
- **Step-by-Step Execution**: The system processes tasks incrementally, adjusting the strategy based on intermediate outputs and feedback.
- **Error Handling**: It detects and handles errors during code execution, making necessary adjustments to strategies.
- **Dynamic Strategy Updates**: Based on each step’s outcome, the system adapts the strategy and ensures that the final model is optimized for the user's task.
- **Markdown Reporting**: All actions, including the generated code, output, and analysis, are appended to a markdown file for user reference.
- **Handles Large Datasets**: Capable of working with large datasets (up to 1GB), ensuring scalability.

## Installation

1. Clone the repository:
    ```bash
    git clone <[your-repository-url](https://github.com/yug201/Data-Scientist-AI-system)>
    
    ```

2. Install required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the environment (ensure that you have a working Python environment and dependencies).

## Usage

1. **Input the Dataset Path and Description**: 
    The system takes the dataset path and description as input. For example:
    ```python
    dataset_path = 'path_to_your_dataset.csv'
    dataset_description = 'Description of the dataset'
    ```

2. **Define the Task**:
    Specify the task you want the system to perform. For example, if you want to predict a column in the dataset, define it:
    ```python
    user_task = 'Predict column_name'
    ```

3. **Run the System**:
    After providing the dataset path, description, and task, the system will automatically generate a strategy, execute each step, handle any errors, and output the results in a markdown file.
    
    Example:
    ```python
    system.run(dataset_path, dataset_description, user_task)
    ```

    This will:
    - Generate a strategy for ML model creation.
    - Execute each step, adjusting based on feedback.
    - Handle errors and retry logic if necessary.
    - Append all code, output, and analysis to a structured markdown file.

4. **Review the Results**:
    After execution, check the markdown file (`file.md`) for a detailed report of each step, including:
    - **Generated Code**: The code for each step.
    - **Output**: The result of each code execution.
    - **Analysis**: Insights, errors, and strategy updates based on each step.

## File Structure

- **`file.md`**: Contains the detailed report, including code, output, and analysis for each step.
- **`file2.txt`**: Stores previous steps’ code and analysis to maintain context for future steps.
- **`file_analysis.txt`**: Contains additional analysis related to each step.
- **`file1.txt`**: Contains initial dataset information and user queries.

## Example Output

Here’s what the final markdown file (`file.md`) may look like:

```markdown
### Step 1: Data Preprocessing
```python
# Generated Code for Data Preprocessing
import pandas as pd

# Load the dataset
df = pd.read_csv('path_to_your_dataset.csv')
df.head()
```

```output
# Output for Data Preprocessing
   column1  column2  column3
0      10      20      30
```

### Analysis for Step 1:
- Data loaded successfully.
- No missing values were found.
- The dataset seems to be ready for ML modeling.

### Step 2: Model Training
```python
# Generated Code for Model Training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Prepare data for modeling
X = df.drop('target_column', axis=1)
y = df['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

```output
# Output for Model Training
Model trained successfully.
```

### Analysis for Step 2:
- The model was successfully trained with a RandomForestClassifier.
- No errors were encountered.
- Proceeding with evaluation.
```

## Error Handling

The system is designed to handle errors automatically. If an error occurs during code execution:
1. The system captures the error message.
2. It adjusts the strategy based on the error, retrying if necessary.
3. The error details and updated strategy are documented in the markdown report for transparency.

For persistent errors (e.g., after multiple attempts), the system will suggest changes to the strategy or advise dropping the task.

## Future Improvements

- **Support for more ML algorithms**: Extend the system to support additional machine learning models.
- **Enhanced error handling**: Further improvements in error detection and handling.
- **Improved visualization**: Adding more detailed visualizations to help the user understand each step.

## Contributing

Contributions are welcome! If you find any bugs or have ideas for improvements, feel free to fork the repository, make changes, and submit a pull request.
