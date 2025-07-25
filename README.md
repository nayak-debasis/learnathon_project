# Auto Insurance Fraud Detection

This repository contains a Python script for detecting fraudulent claims in auto insurance datasets. The solution involves robust data preprocessing, insightful feature engineering, and the application of a powerful Random Forest Classifier to accurately predict the likelihood of fraud. This automated approach aims to enhance the efficiency and effectiveness of fraud detection processes within the insurance industry.

## Table of Contents

* [Project Overview](https://www.google.com/search?q=%23project-overview)

* [Dataset](https://www.google.com/search?q=%23dataset)

* [Features](https://www.google.com/search?q=%23features)

* [Preprocessing Steps](https://www.google.com/search?q=%23preprocessing-steps)

* [Model](https://www.google.com/search?q=%23model)

* [Evaluation Metrics](https://www.google.com/search?q=%23evaluation-metrics)

* [Results](https://www.google.com/search?q=%23results)

* [Feature Importance](https://www.google.com/search?q=%23feature-importance)

* [Usage](https://www.google.com/search?q=%23usage)

* [Requirements](https://www.google.com/search?q=%23requirements)

## Project Overview

The primary objective of this project is to develop a sophisticated machine learning model capable of identifying fraudulent auto insurance claims. By meticulously analyzing historical claim data, the model is designed to learn intricate patterns and subtle anomalies that are characteristic of fraudulent activities. This proactive identification of suspicious claims can significantly reduce financial losses for insurance companies, streamline the claims processing workflow, and deter potential fraudsters. Ultimately, this project contributes to a more secure and equitable insurance ecosystem by ensuring that legitimate claims are processed swiftly while fraudulent ones are flagged for further investigation.

## Dataset

The project leverages a comprehensive dataset distributed across three distinct CSV files: `Auto_Insurance_Fraud_Claims_File01.csv`, `Auto_Insurance_Fraud_Claims_File02.csv`, and `Auto_Insurance_Fraud_Claims_File03.csv`. These files collectively contain a wide array of auto insurance claim data. Additionally, a separate submission file, `Auto_Insurance_Fraud_Claims_Results_Submission.csv`, is utilized specifically to provide the crucial `Fraud_Ind` (Fraud Indicator) column, which serves as our target variable. While the exact origin of the data (e.g., simulated, anonymized real-world) is not specified, it is structured to represent typical insurance claim records, encompassing various attributes pertaining to policyholders, accident circumstances, vehicle specifications, and detailed claim financial information.

## Features

The dataset is rich with numerous features, each providing valuable information for the fraud detection task. Here's a detailed look at some of the key attributes:

* `Claim_ID`: A unique alphanumeric identifier assigned to each individual insurance claim, essential for tracking and merging data.

* `Bind_Date1`: The date when the insurance policy was bound or activated. This date is crucial for calculating policy duration and other time-based features.

* `Customer_Life_Value1`: An estimated value representing the long-term profitability of a customer to the insurance company. This can indicate the potential impact of a fraudulent claim from a high-value customer.

* `Age_Insured`: The age of the insured individual at the time of the claim. Age can sometimes correlate with certain types of claim behavior.

* `Policy_Num`: The unique policy number associated with the insurance coverage.

* `Policy_State`: The state where the insurance policy was issued. Geographic location can be a significant factor in fraud patterns due to varying regulations or regional crime rates.

* `Policy_Start_Date`, `Policy_Expiry_Date`: These dates define the active period of the insurance policy. They are vital for calculating the duration of coverage and the time elapsed since policy inception.

* `Policy_BI`, `Policy_Ded`, `Policy_Premium`: These represent critical policy details: Bodily Injury liability limits, the deductible amount the insured must pay, and the premium paid for the policy. These financial terms can influence claim behavior.

* `Umbrella_Limit`: The limit of an umbrella liability policy, which provides additional coverage beyond standard policies. High umbrella limits might attract different types of fraudulent attempts.

* `Insured_Zip`: The postal code of the insured's residence, offering another geographical data point.

* `Gender`, `Education`, `Occupation`, `Hobbies`, `Insured_Relationship`: These demographic and personal details of the insured can provide insights into behavioral patterns and risk profiles. For example, certain occupations or hobbies might be associated with higher or lower risk of accidents or fraud.

* `Capital_Gains`, `Capital_Loss`: Financial information related to the insured's capital gains and losses. Sudden financial distress (indicated by significant capital loss) could potentially be a motivator for fraud.

* `Garage_Location`: A categorical indicator of whether the vehicle is typically garaged. This might influence the likelihood of certain types of claims, such as theft.

* `Accident_Date`, `Accident_Type`, `Collision_Type`, `Accident_Severity`: Detailed information about the accident itself, including its date, the nature of the accident (e.g., single vehicle, multi-vehicle), the type of collision (e.g., rear, front, side), and the severity of the damage (e.g., total loss, major damage, minor damage, trivial damage). These are often primary indicators of claim legitimacy.

* `authorities_contacted`: Indicates whether law enforcement or emergency services (e.g., Police, Ambulance, Fire) were contacted after the accident. The absence or presence of official reports can be a red flag.

* `Acccident_State`, `Acccident_City`, `Accident_Location`: Specific geographical details of where the accident occurred, which can be cross-referenced with policy state and insured zip code for consistency checks.

* `Accident_Hour`: The hour of the day when the accident took place. Certain hours might have higher incidences of specific accident types or fraud.

* `Num_of_Vehicles_Involved`, `Property_Damage`, `Bodily_Injuries`, `Witnesses`, `Police_Report`: These features describe the immediate impact and aftermath of the accident, including the number of vehicles, whether property damage or bodily injuries occurred, the presence of witnesses, and if a police report was filed. The consistency of these details is crucial for fraud detection.

* `DL_Expiry_Date`, `Claims_Date`: The expiry date of the driver's license and the date the claim was filed. Timeliness of claim filing relative to the accident date can be a factor.

* `Auto_Make`, `Auto_Model`, `Auto_Year`, `Vehicle_Color`, `Vehicle_Cost`, `Annual_Mileage`, `DiffIN_Mileage`: Comprehensive details about the insured vehicle, including its make, model, year, color, estimated cost, annual mileage, and the difference in mileage. Discrepancies in vehicle details or unusually high/low mileage changes can be suspicious.

* `Low_Mileage_Discount`, `Commute_Discount`: Indicators of specific discounts applied to the policy.

* `Total_Claim`, `Injury_Claim`, `Property_Claim`, `Vehicle_Claim`: The monetary amounts claimed for total damages, injuries, property damage, and vehicle damage. Unusually high claims for minor accidents, or disproportionate claim components, can suggest fraud.

* `Vehicle_Registration`: The vehicle's registration number, another unique identifier.

* `Check_Point`: An additional, unspecified check point.

* `Fraud_Ind`: The crucial target variable, indicating whether a claim is fraudulent ('Y') or non-fraudulent ('N'). This is the label the model aims to predict.

## Preprocessing Steps

Thorough data preprocessing is paramount for building an effective machine learning model. The following steps ensure the data is clean, consistent, and in a suitable format for model training:

1. **Data Loading and Merging**:

   * The initial phase involves loading the three individual claim data CSV files (`Auto_Insurance_Fraud_Claims_File01.csv`, `Auto_Insurance_Fraud_Claims_File02.csv`, and `Auto_Insurance_Fraud_Claims_File03.csv`) into Pandas DataFrames.

   * These DataFrames are then concatenated vertically (`pd.concat`) to form a single, unified dataset, preserving all original records.

   * Crucially, the `Auto_Insurance_Fraud_Claims_Results_Submission.csv` file, which contains the ground truth for fraud (the `Fraud_Ind` column), is loaded. This `Fraud_Ind` column is then merged into the combined claims DataFrame using `Claim_ID` as the common key. This step is vital as it associates each claim with its known fraud status, allowing for supervised learning.

2. **Date Conversion**:

   * Date-related columns (e.g., `Bind_Date1`, `Policy_Start_Date`, `Accident_Date`, `DL_Expiry_Date`, `Claims_Date`) are initially loaded as strings or objects. To enable time-based calculations and feature engineering, these columns are systematically converted into `datetime` objects using `pd.to_datetime`. The `errors='coerce'` argument is used to handle any unparseable date strings by converting them to `NaT` (Not a Time), preventing the script from crashing.

3. **Handling Missing Values**:

   * Missing data, if not addressed, can lead to biased models and errors. For numerical columns (e.g., `Age_Insured`, `Total_Claim`), missing values are imputed with their respective *median* values. The median is chosen over the mean to minimize the impact of outliers.

   * For categorical columns (e.g., `Gender`, `Occupation`), missing values are imputed with the *mode* (most frequent category). This strategy assumes that the most common category is a reasonable substitute for missing entries.

   * A specific handling is applied for 'Unknown' values within the `Property_Damage` column, replacing them with the mode of that column. This ensures consistency and prevents 'Unknown' from being treated as a distinct, meaningful category if it's merely a placeholder for missing information.

4. **Feature Engineering**:

   * This step involves creating new, potentially more informative features from existing ones.

   * For all converted datetime columns, new numerical features are extracted: `_Year`, `_Month`, and `_Day`. For example, `Accident_Date` will yield `Accident_Date_Year`, `Accident_Date_Month`, and `Accident_Date_Day`. These granular time components can capture seasonal or temporal patterns related to fraud.

   * `Vehicle_Age_at_Accident` is calculated by subtracting `Auto_Year` from `Accident_Date_Year`. This feature provides a direct measure of the vehicle's age at the time of the incident, which can be a risk factor. Missing values in this new feature are imputed with the median.

   * `Days_Policy_to_Accident` is computed as the difference in days between `Accident_Date` and `Policy_Start_Date`. This feature indicates how long the policy had been active before the accident occurred, which can sometimes reveal suspicious patterns (e.g., claims filed very soon after policy inception). Missing values are also imputed with the median.

5. **Target Variable Encoding**:

   * The `Fraud_Ind` column, which is initially categorical ('Y' or 'N'), needs to be converted into a numerical format for machine learning algorithms. 'Y' (fraudulent) is mapped to `1`, and 'N' (non-fraudulent) is mapped to `0`. Any remaining missing values in this target column are imputed with its mode, ensuring a complete numerical target variable.

6. **Column Dropping**:

   * Certain columns are removed from the dataset. This includes the original date columns (as their extracted components are now available), and identifier columns like `Claim_ID`, `Policy_Num`, `Vehicle_Registration`, and `Check_Point`. These identifiers are typically not useful for predictive modeling and can sometimes introduce noise or leakage if not handled carefully. Removing them simplifies the model and reduces dimensionality.

7. **Categorical Encoding**:

   * All remaining categorical columns (e.g., `Policy_State`, `Accident_Type`, `Gender`, `Occupation`) are converted into a numerical format using one-hot encoding via `pd.get_dummies`. This process creates new binary (0 or 1) columns for each unique category within a feature. For instance, `Policy_State` might become `Policy_State_OH`, `Policy_State_IL`, etc. The `drop_first=True` argument is employed to prevent multicollinearity, where one category can be inferred from the others, which can sometimes cause issues in certain models.

## Model

A **Random Forest Classifier** is chosen as the core machine learning model for this fraud detection task. Random Forest is an ensemble learning method that operates by constructing a multitude of decision trees during training and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. It is well-suited for this problem due to its ability to handle high-dimensional data, capture non-linear relationships, and provide good predictive performance while being relatively robust to outliers and noise.

The specific hyperparameters used are:

* `n_estimators=100`: This parameter specifies the number of decision trees in the forest. A higher number generally leads to more stable and accurate predictions but also increases computation time. 100 trees provide a good balance between performance and efficiency for many datasets.

* `random_state=42`: Setting a `random_state` ensures that the random processes involved in building the trees (e.g., feature selection, data sampling) are reproducible. This means that if you run the code multiple times, you will get the exact same results, which is crucial for debugging and comparing models.

* `class_weight='balanced'`: This is a critical parameter for imbalanced datasets, like fraud detection where fraudulent claims are typically a small minority. By setting `class_weight='balanced'`, the model automatically adjusts the weights inversely proportional to class frequencies. This gives more importance to the minority class (fraudulent claims) during training, helping the model to learn patterns from fewer examples and preventing it from simply predicting the majority class all the time. Without this, a model might achieve high overall accuracy by correctly classifying all non-fraudulent cases but completely miss all fraudulent ones.

## Evaluation Metrics

To thoroughly assess the model's effectiveness in identifying fraud, a suite of evaluation metrics is employed. Each metric provides a different perspective on performance, which is especially important in imbalanced classification problems like fraud detection:

* **Accuracy**: Represents the proportion of total predictions that were correct. While intuitive, accuracy can be misleading in imbalanced datasets. For example, if only 1% of claims are fraudulent, a model that always predicts "non-fraudulent" would achieve 99% accuracy, but would be useless for fraud detection.

* **Precision**: Measures the proportion of true positive predictions (correctly identified fraudulent claims) among all positive predictions made by the model (all claims predicted as fraudulent). High precision means fewer false positives, which is important if investigating flagged claims is costly.

* **Recall (Sensitivity)**: Measures the proportion of true positive predictions among all actual positive instances (all truly fraudulent claims). High recall means fewer false negatives, which is crucial in fraud detection to avoid missing actual fraudulent activities. In many fraud scenarios, missing a fraudulent claim (false negative) is more costly than investigating a legitimate one (false positive).

* **F1-Score**: The harmonic mean of precision and recall. It provides a single metric that balances both precision and recall, offering a more robust measure of a model's performance on imbalanced datasets than accuracy alone.

* **ROC AUC (Receiver Operating Characteristic Area Under the Curve)**: This metric evaluates the model's ability to distinguish between the two classes (fraudulent vs. non-fraudulent) across various classification thresholds. A higher ROC AUC score indicates a better ability to separate the classes. It is particularly useful for imbalanced datasets as it is less sensitive to class distribution.

* **Classification Report**: A comprehensive text summary that provides precision, recall, F1-score, and support (number of actual occurrences) for each class (0: non-fraudulent, 1: fraudulent). This report offers a quick overview of class-specific performance.

* **Confusion Matrix**: A table that visually summarizes the performance of a classification algorithm. It shows the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).

  * TP: Correctly predicted fraudulent claims.

  * TN: Correctly predicted non-fraudulent claims.

  * FP: Non-fraudulent claims incorrectly predicted as fraudulent (Type I error).

  * FN: Fraudulent claims incorrectly predicted as non-fraudulent (Type II error).
    The confusion matrix is invaluable for understanding where the model is succeeding and where it is failing, especially regarding the trade-off between false positives and false negatives.

## Results

The model training and evaluation process generates detailed performance metrics, providing insights into its effectiveness. Below is an example output from a typical run:

--- Model Training and Evaluation Complete ---
Model Used: Random Forest Classifier
Accuracy: 0.8571
Precision: 0.0000
Recall: 0.0000
F1-Score: 0.0000
ROC AUC: 0.0000

Classification Report:

          precision    recall  f1-score   support

       0       0.86      1.00      0.92         6
       1       0.00      0.00      0.00         1

accuracy                           0.86         7
macro avg       0.43      0.50      0.46         7
weighted avg       0.73      0.86      0.79         7

Confusion Matrix:

[[6 0]
[1 0]]


**Note on Precision/Recall/F1-Score for Class 1 (Fraudulent):**
The output indicates that for the fraudulent class (class 1), the precision, recall, and F1-score are all 0.00. This is a critical observation. It signifies that in this particular test set, the model did not correctly identify any fraudulent claims, or it did not predict any samples as fraudulent at all. This often occurs in scenarios with severe class imbalance, where the fraudulent class is extremely rare (as indicated by `support` of 1 for class 1 vs. 6 for class 0 in this small example). Despite using `class_weight='balanced'`, the model might still struggle to learn from very few positive examples and default to predicting the more prevalent majority class (non-fraudulent).

To improve performance on the minority class, especially recall, further advanced techniques might be necessary. These could include:

* **Oversampling minority class**: Techniques like SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic samples of the minority class.

* **Undersampling majority class**: Reducing the number of samples in the majority class.

* **Different ensemble methods**: Exploring other algorithms or more sophisticated stacking/boosting techniques.

* **Cost-sensitive learning**: Explicitly assigning higher misclassification costs to false negatives.

* **Anomaly detection approaches**: Treating fraud detection as an anomaly detection problem rather than pure classification.

* **Feature engineering refinement**: Creating even more powerful features that distinctly separate fraudulent from non-fraudulent cases.

## Feature Importance

Feature importance quantifies the contribution of each input feature to the model's predictions. Understanding feature importance is invaluable for several reasons: it helps in interpreting the model, identifying key drivers of fraud, and potentially guiding further data collection or domain-specific investigations. The top 10 most important features, as determined by the Random Forest model, are:

Top 10 Feature Importances:

Days_Policy_to_Accident     0.089738
Policy_Expiry_Date_Month    0.076575
Policy_Start_Date_Month     0.057962
Total_Claim                 0.044705
Accident_Date_Month         0.042482
Property_Claim              0.041731
Annual_Mileage              0.032332
DiffIN_Mileage              0.031816
Capital_Gains               0.030121
Acccident_State_SC          0.029619
dtype: float64


These features are identified as having the most significant impact on whether a claim is classified as fraudulent or not. For instance, `Days_Policy_to_Accident` being a top feature suggests that the time elapsed between policy inception and the accident date is a strong indicator. Similarly, various date-related features (`Policy_Expiry_Date_Month`, `Policy_Start_Date_Month`, `Accident_Date_Month`) imply that seasonality or specific timing patterns might be linked to fraudulent activities. Financial aspects like `Total_Claim`, `Property_Claim`, and `Capital_Gains` also play a crucial role, indicating that monetary values and financial background are strong predictors. This information can be used by insurance investigators to focus their efforts on claims exhibiting these characteristics.

## Usage

To effectively utilize and run this auto insurance fraud detection script, follow these steps:

1. **Save the Code**: Begin by saving the provided Python code. You can either save it as a standard Python file (e.g., `fraud_detection.py`) or, for an interactive experience, paste it directly into a Jupyter Notebook or Google Colab environment. Using a notebook environment can be beneficial for step-by-step execution and visualization of intermediate results.

2. **Place Data Files**: Ensure that all necessary input data files are accessible to your script. Specifically, the three claim data files (`Auto_Insurance_Fraud_Claims_File01.csv`, `Auto_Insurance_Fraud_Claims_File02.csv`, and `Auto_Insurance_Fraud_Claims_File03.csv`) and the submission file (`Auto_Insurance_Fraud_Claims_Results_Submission.csv`) must be located in the same directory as your Python script or notebook. If your data is in a different location, you will need to update the file paths in the script accordingly.

3. **Install Dependencies**: Before running the script, make sure you have all the required Python libraries installed in your environment. The list of dependencies is provided in the [Requirements](https://www.google.com/search?q=%23requirements) section below. It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects.

4. **Run the Script**:

   * **From Terminal (for `.py` file)**: Navigate to the directory where you saved `fraud_detection.py` using your command line interface (CLI) and execute the script with the Python interpreter:

     ```
     python fraud_detection.py
     
     ```

   * **In Jupyter Notebook/Colab**: If you are using a notebook environment, simply run each code cell sequentially. This allows you to inspect variables and outputs at each stage of the preprocessing and modeling pipeline.

Upon successful execution, the script will print the comprehensive model evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC), a detailed Classification Report, the Confusion Matrix, and the top 10 Feature Importances directly to your console or notebook output. These outputs provide immediate insights into the model's performance and the factors driving its predictions.

## Requirements

The following Python libraries are essential for running this fraud detection script. These libraries provide the necessary functionalities for data manipulation, machine learning, and visualization:

* `pandas`: A powerful library for data manipulation and analysis, particularly for working with tabular data (DataFrames).

* `scikit-learn`: A widely used machine learning library that provides various classification algorithms (like Random Forest), preprocessing tools, and evaluation metrics.

* `numpy`: The fundamental package for numerical computing in Python, essential for handling arrays and mathematical operations.

* `matplotlib`: A foundational plotting library for creating static, interactive, and animated visualizations in Python.

* `seaborn`: A statistical data visualization library based on matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.

You can easily install all these required libraries using pip, the Python package installer. Open your terminal or command prompt and execute the following command:

pip install pandas scikit-learn numpy matplotlib seaborn
