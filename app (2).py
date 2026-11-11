import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="UAP Academic Performance", page_icon="📊", layout="wide")
st.title("UAP Students Academic Performance Report")
st.write(
    "Descriptive statistics, hypothesis testing, and grade prediction "
    "using Ridge Regression and Random Forest."
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Midterm Grade Survey.csv")

try:
    datos = load_data()
except FileNotFoundError:
    st.error(
        "The file 'Midterm Grade Survey.csv' was not found. "
        "Save the CSV file in the same folder as app.py."
    )
    st.stop()

st.success(f"Dataset loaded: {len(datos)} survey responses and {len(datos.columns)} original variables.")

# Exact column names from the current survey
student_type = "What kind of student are you?"
school = "What school is your career part of?"
subjects = "How many subjects are you currently taking?\n(If more than 10 select 10)"
study_hours = "On average, how many hours per week do you spend studying outside of class?\n(Example: 6.5 hours)"
study_method = "Select your primary study method for midterm exam preparation"
midterm_grade = "What was your average grade percentage on your recent midterm exams?"
gpa_column = "What was your GPA from last semester? \nExample: 9.3 (No special characters)"
scholarship = "Do you have scholarship or any financial education  support"
study_start = "When do you usually start studying for your midterm exams?"
routine_affected = "To what extent do you feel your routine is affected by midterm exams?\n(1 being the lowest and 5 the highest)"
regular_sleep = "On average, how many hours of sleep do you get per night during a regular academic week?\n"
midterm_sleep = "On average, how many hours of sleep did you get on midterm week?"
stress = "How would you rate your overall stress level during the midterm examination period?\n(1 being the lowest and 5 the highest)"
self_rated = "In a scale of 1 to 10, how was your performance on your midterm exams?\n(1 being the lowest and 10 the highest)"
employment = "Are you currently employed while attending classes?"
extracurricular = "Do you participate on extracurricular activities, student committees, or leadership roles?"
financial_support = "What is your primary source of financial support for tuition and living expenses?"
income = "What is your income per month?\n(Example: 10000) Type only number, no special characters or commas. "

# Check that the expected survey columns exist
required_columns = [
    student_type, school, subjects, study_hours, study_method, midterm_grade,
    gpa_column, scholarship, study_start, routine_affected, regular_sleep,
    midterm_sleep, stress, self_rated, employment, extracurricular,
    financial_support, income
]
missing_columns = [col for col in required_columns if col not in datos.columns]
if missing_columns:
    st.error("Some expected survey columns were not found. Check that you are using the updated CSV file.")
    st.write("Missing columns:", missing_columns)
    st.stop()

# ---------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------
def number_from_answer(value):
    """Convert answers such as '4 hours' or '3-4' into a numeric value."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace(",", ".")
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if not numbers:
        return np.nan
    values = [float(number) for number in numbers]
    # If the answer is a range such as 3-4 or 2 a 3, use its midpoint.
    if len(values) >= 2 and ("-" in text or " a " in text or " to " in text):
        return float(np.mean(values[:2]))
    return values[0]

# GPA is the prediction target. A few answers were entered as 91 or 92;
# these are interpreted as 9.1 and 9.2 on the UAP 0-10 scale.
datos["GPA_clean"] = pd.to_numeric(datos[gpa_column].astype(str).str.strip(), errors="coerce")
datos.loc[datos["GPA_clean"] > 10, "GPA_clean"] = datos.loc[datos["GPA_clean"] > 10, "GPA_clean"] / 10
datos.loc[~datos["GPA_clean"].between(0, 10), "GPA_clean"] = np.nan

# Clean numeric answers that were entered as text
datos["Study_hours_clean"] = datos[study_hours].apply(number_from_answer)

# Income answers sometimes contain thousands separators, for example "10,000".
def income_to_number(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return float(numbers[0]) if numbers else np.nan

datos["Income_clean"] = datos[income].apply(income_to_number)

# Normalize categorical answers
def clean_category(series):
    return series.astype("string").str.strip().replace("", pd.NA)

for col in [student_type, school, study_method, scholarship, study_start,
            employment, extracurricular, financial_support]:
    datos[col] = clean_category(datos[col])

# ---------------------------------------------------------
# VARIABLE GROUPS
# ---------------------------------------------------------
st.header("1. Variable Groups")
st.write(
    "The survey variables are organized into three groups. This organization "
    "helps explain the research questions and identify possible predictors."
)

performance_group = {
    "Academic performance": [
        "GPA_clean",
        midterm_grade,
        self_rated,
    ]
}
habits_group = {
    "Academic habits and lifestyle": [
        subjects,
        "Study_hours_clean",
        study_method,
        study_start,
        regular_sleep,
        midterm_sleep,
        stress,
        routine_affected,
    ]
}
student_group = {
    "Student characteristics": [
        student_type,
        school,
        scholarship,
        employment,
        extracurricular,
        financial_support,
        "Income_clean",
    ]
}

with st.expander("Group 1 – Academic performance", expanded=True):
    st.write("Variables directly related to students' academic results.")
    st.dataframe(pd.DataFrame({"Variable": ["Previous semester GPA", "Average recent midterm grade", "Self-rated midterm performance"]}), use_container_width=True)
with st.expander("Group 2 – Academic habits and lifestyle", expanded=True):
    st.write("Variables related to study routines, sleep, workload, and stress.")
    st.dataframe(pd.DataFrame({"Variable": ["Number of subjects", "Weekly study hours", "Study method", "When studying begins", "Regular-week sleep", "Midterm-week sleep", "Stress level", "Routine affected"]}), use_container_width=True)
with st.expander("Group 3 – Student characteristics", expanded=True):
    st.write("Variables describing students' academic and financial context.")
    st.dataframe(pd.DataFrame({"Variable": ["Student type", "School", "Scholarship", "Employment", "Extracurricular activities", "Financial support", "Monthly income"]}), use_container_width=True)

st.caption(
    "The groups organize the analysis; they do not automatically prove that a variable is statistically important. "
    "Statistical tests and model evaluation are used for different purposes."
)

# ---------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------
st.header("2. Descriptive Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Survey responses", len(datos))
col2.metric("Valid previous GPA values", int(datos["GPA_clean"].notna().sum()))
col3.metric("Variables in survey", len(datos.columns) - 3)

st.subheader("Academic performance")
performance_data = datos[["GPA_clean", midterm_grade, self_rated]].copy()
performance_data.columns = ["Previous semester GPA", "Average midterm grade", "Self-rated midterm performance"]
st.dataframe(performance_data.describe().T, use_container_width=True)

st.subheader("Academic habits and lifestyle")
habits_data = datos[[subjects, "Study_hours_clean", regular_sleep, midterm_sleep, stress, routine_affected]].copy()
habits_data.columns = ["Number of subjects", "Study hours per week", "Regular sleep hours", "Midterm sleep hours", "Stress level", "Routine affected"]
st.dataframe(habits_data.describe().T, use_container_width=True)

st.subheader("Student characteristics")
characteristics = pd.DataFrame({
    "Student type": datos[student_type].value_counts(dropna=False),
    "Scholarship": datos[scholarship].value_counts(dropna=False),
    "Employment": datos[employment].value_counts(dropna=False),
})
st.dataframe(characteristics.fillna(0), use_container_width=True)

# ---------------------------------------------------------
# 3. DATA VISUALIZATION
# ---------------------------------------------------------
st.header("3. Data Visualization")
plot_col1, plot_col2 = st.columns(2)
with plot_col1:
    fig, ax = plt.subplots()
    ax.bar(["Regular week", "Midterm week"], [datos[regular_sleep].mean(), datos[midterm_sleep].mean()])
    ax.set_ylabel("Average hours of sleep")
    ax.set_title("Average sleep hours comparison")
    st.pyplot(fig)
    plt.close(fig)
with plot_col2:
    fig, ax = plt.subplots()
    datos["GPA_clean"].dropna().hist(ax=ax, bins=10)
    ax.set_xlabel("Previous semester GPA")
    ax.set_ylabel("Number of students")
    ax.set_title("Distribution of previous semester GPA")
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------
# 4. PAIRED HYPOTHESIS TESTS: SLEEP
# ---------------------------------------------------------
st.header("4. Hypothesis Testing: Sleep Hours")
sleep_data = datos[[regular_sleep, midterm_sleep]].apply(pd.to_numeric, errors="coerce").dropna()
x = sleep_data[regular_sleep]
y = sleep_data[midterm_sleep]
differences = x - y
alpha = 0.05

st.subheader("Research question")
st.write("Do students sleep a different number of hours during a regular academic week compared with midterm week?")
st.write("**H0:** The paired sleep-hour differences are centered at zero.")
st.write("**H1:** The paired sleep-hour differences are not centered at zero.")
st.write(f"Significance level: {alpha}")
st.write(f"Students with answers to both questions: {len(sleep_data)}")
st.write(f"Mean sleep hours during a regular week: {x.mean():.2f}")
st.write(f"Mean sleep hours during midterm week: {y.mean():.2f}")

# Paired t-test
st.subheader("4.1 Paired T-Test (Parametric)")
t_stat, p_ttest = stats.ttest_rel(x, y)
st.write("**H0:** The population mean paired difference is zero.")
st.write("**H1:** The population mean paired difference is not zero.")
st.write(f"T statistic: {t_stat:.4f}")
st.write(f"P-value: {p_ttest:.4f}")
if p_ttest < alpha:
    st.success("Decision: Reject H0. There is evidence of a difference in mean sleep hours.")
else:
    st.info("Decision: Do not reject H0. There is not enough evidence of a difference in mean sleep hours.")

# Sign test
st.subheader("4.2 Sign Test (Non-Parametric)")
nonzero_differences = differences[differences != 0]
positive = int((nonzero_differences > 0).sum())
negative = int((nonzero_differences < 0).sum())
if len(nonzero_differences) > 0:
    sign_result = stats.binomtest(positive, n=len(nonzero_differences), p=0.5, alternative="two-sided")
    p_sign = sign_result.pvalue
    st.write("**H0:** Positive and negative paired differences are equally likely (median difference = 0).")
    st.write("**H1:** Positive and negative paired differences are not equally likely (median difference ≠ 0).")
    st.write(f"Positive differences (regular sleep > midterm sleep): {positive}")
    st.write(f"Negative differences (regular sleep < midterm sleep): {negative}")
    st.write(f"Zero differences excluded: {int((differences == 0).sum())}")
    st.write(f"P-value: {p_sign:.4f}")
    if p_sign < alpha:
        st.success("Decision: Reject H0. The signs of the paired differences are not equally distributed.")
    else:
        st.info("Decision: Do not reject H0. There is not enough evidence of an imbalance in the signs.")
else:
    p_sign = np.nan
    st.warning("The Sign Test cannot be calculated because all paired differences are zero.")

# Wilcoxon signed-rank test
st.subheader("4.3 Wilcoxon Signed-Rank Test (Non-Parametric)")
st.write("**H0:** The distribution of paired sleep-hour differences is centered at zero.")
st.write("**H1:** The distribution of paired sleep-hour differences is not centered at zero.")
try:
    if len(nonzero_differences) > 0:
        w_stat, p_wilcoxon = stats.wilcoxon(x, y, alternative="two-sided")
        st.write(f"Wilcoxon statistic: {w_stat:.4f}")
        st.write(f"P-value: {p_wilcoxon:.4f}")
        if p_wilcoxon < alpha:
            st.success("Decision: Reject H0. There is evidence of a difference in paired sleep hours.")
        else:
            st.info("Decision: Do not reject H0. There is not enough evidence of a difference in paired sleep hours.")
    else:
        p_wilcoxon = np.nan
        st.warning("The Wilcoxon test cannot be calculated because all paired differences are zero.")
except ValueError as error:
    p_wilcoxon = np.nan
    st.warning(f"Wilcoxon test could not be calculated: {error}")

# ---------------------------------------------------------
# 5. CHI-SQUARE TEST OF INDEPENDENCE
# ---------------------------------------------------------
st.header("5. Chi-Square Test of Independence")
st.write("Research question: Is student employment associated with having a scholarship or financial education support?")

# Combine part-time and full-time employment into one category to create a 2x2 table.
chi_data = datos[[employment, scholarship]].dropna().copy()
chi_data["Employment status"] = np.where(chi_data[employment].str.strip().eq("No"), "Not employed", "Employed")
chi_data["Scholarship status"] = chi_data[scholarship].str.strip()
contingency_table = pd.crosstab(chi_data["Employment status"], chi_data["Scholarship status"])

st.write("**H0:** Employment status and scholarship status are independent (there is no association).")
st.write("**H1:** Employment status and scholarship status are not independent (there is an association).")
st.write("Observed frequencies:")
st.dataframe(contingency_table, use_container_width=True)

if contingency_table.shape[0] >= 2 and contingency_table.shape[1] >= 2:
    chi_stat, chi_p, chi_df, expected = stats.chi2_contingency(contingency_table, correction=False)
    st.write(f"Chi-square statistic: {chi_stat:.4f}")
    st.write(f"Degrees of freedom: {chi_df}")
    st.write(f"P-value: {chi_p:.4f}")
    st.write(f"Minimum expected frequency: {expected.min():.2f}")
    if expected.min() < 5:
        st.warning("At least one expected frequency is below 5. Interpret the chi-square approximation with caution; consider combining justified categories or using an exact test.")
    if chi_p < alpha:
        st.success("Decision: Reject H0. There is evidence of an association between employment and scholarship status.")
    else:
        st.info("Decision: Do not reject H0. There is not enough evidence of an association between employment and scholarship status.")
else:
    st.warning("The selected categorical variables do not form a table with at least two rows and two columns.")

# ---------------------------------------------------------
# 6. HYPOTHESIS TEST FOR GRADES
# ---------------------------------------------------------
st.header("6. Hypothesis Testing for Grades")
st.subheader("One-Sample T-Test")
gpa_values = datos["GPA_clean"].dropna()
reference_gpa = 8.0

st.write("Research question: Is the average previous-semester GPA different from 8.0?")
st.write("**H0:** The population mean previous-semester GPA is equal to 8.0.")
st.write("**H1:** The population mean previous-semester GPA is different from 8.0.")
st.write(f"Reference GPA: {reference_gpa:.1f}")
st.write(f"Valid GPA responses: {len(gpa_values)}")
if len(gpa_values) >= 2:
    grade_t, grade_p = stats.ttest_1samp(gpa_values, popmean=reference_gpa)
    st.write(f"Sample mean GPA: {gpa_values.mean():.3f}")
    st.write(f"T statistic: {grade_t:.4f}")
    st.write(f"P-value: {grade_p:.4f}")
    if grade_p < alpha:
        st.success("Decision: Reject H0. The sample provides evidence that the population mean GPA differs from 8.0.")
    else:
        st.info("Decision: Do not reject H0. There is not enough evidence that the population mean GPA differs from 8.0.")
else:
    st.warning("There are not enough valid GPA values to perform the test.")

st.caption("The one-sample t-test assumes independent observations and an approximately normal sampling distribution of the mean. Interpret results in the context of the survey sample.")

# ---------------------------------------------------------
# 7. MACHINE LEARNING: RIDGE AND RANDOM FOREST
# ---------------------------------------------------------
st.header("7. Grade Prediction Models")
st.write(
    "The target variable is the GPA from the previous semester. Ridge Regression "
    "and Random Forest Regression are trained and evaluated on a held-out test set."
)
st.info(
    "Important limitation: this is a retrospective prediction exercise. The survey was collected during the current midterm period, "
    "while the target is the previous-semester GPA. Therefore, the models estimate patterns in this dataset; they do not prove that "
    "these variables would have been available to predict that GPA in real time."
)

# Use habits and student-context variables as predictors.
# GPA and other current grade outcomes are deliberately excluded from the predictors.
numeric_features = [subjects, "Study_hours_clean", routine_affected, regular_sleep, midterm_sleep, stress, "Income_clean"]
categorical_features = [student_type, school, study_method, scholarship, study_start, employment, extracurricular, financial_support]
feature_columns = numeric_features + categorical_features

model_data = datos[feature_columns + ["GPA_clean"]].copy()
model_data = model_data.dropna(subset=["GPA_clean"])
X = model_data[feature_columns].copy()
y_model = model_data["GPA_clean"].copy()

# Convert numeric predictors and leave missing values for the imputer.
for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")
for col in categorical_features:
    X[col] = X[col].astype("string").fillna("Unknown").astype(str)

if len(model_data) < 20:
    st.warning("There are too few complete target observations to train and evaluate the models reliably.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_model, test_size=0.20, random_state=42
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ])

    ridge_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=1.0))
    ])
    forest_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2))
    ])

    ridge_model.fit(X_train, y_train)
    forest_model.fit(X_train, y_train)
    ridge_predictions = ridge_model.predict(X_test)
    forest_predictions = forest_model.predict(X_test)

    def model_metrics(actual, predicted):
        return {
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "R-squared": r2_score(actual, predicted),
        }

    ridge_results = model_metrics(y_test, ridge_predictions)
    forest_results = model_metrics(y_test, forest_predictions)
    results_table = pd.DataFrame(
        [ridge_results, forest_results],
        index=["Ridge Regression", "Random Forest"]
    )

    st.subheader("7.1 Model performance")
    st.write("The models are evaluated using the same 20% test sample.")
    st.dataframe(results_table.style.format("{:.3f}"), use_container_width=True)
    st.caption("Lower MAE and RMSE indicate smaller prediction errors. Higher R-squared indicates more variation explained; a negative R-squared is possible and means the model performs worse than predicting the test-set mean.")

    # Actual vs predicted plot
    st.subheader("7.2 Actual vs. predicted GPA")
    fig, ax = plt.subplots()
    ax.scatter(y_test, ridge_predictions, label="Ridge", alpha=0.75)
    ax.scatter(y_test, forest_predictions, label="Random Forest", alpha=0.75)
    low = min(float(y_test.min()), float(ridge_predictions.min()), float(forest_predictions.min()))
    high = max(float(y_test.max()), float(ridge_predictions.max()), float(forest_predictions.max()))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual previous-semester GPA")
    ax.set_ylabel("Predicted GPA")
    ax.set_title("Actual vs. predicted GPA")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # Display a few held-out predictions
    st.subheader("7.3 Test-set predictions")
    prediction_table = pd.DataFrame({
        "Actual GPA": y_test.values,
        "Ridge prediction": ridge_predictions,
        "Random Forest prediction": forest_predictions,
    }, index=y_test.index)
    st.dataframe(prediction_table.round(2), use_container_width=True)

    # Ridge coefficients and Random Forest feature importance
    st.subheader("7.4 Variables used by the models")
    fitted_preprocessor = forest_model.named_steps["preprocessor"]
    encoded_names = fitted_preprocessor.get_feature_names_out()
    importances = forest_model.named_steps["model"].feature_importances_
    importance_table = pd.DataFrame({
        "Variable": encoded_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(12)
    st.write("Top 12 encoded features according to Random Forest importance:")
    st.dataframe(importance_table, use_container_width=True)
    fig, ax = plt.subplots()
    plot_importance = importance_table.sort_values("Importance")
    ax.barh(plot_importance["Variable"], plot_importance["Importance"])
    ax.set_xlabel("Feature importance")
    ax.set_title("Random Forest: most important encoded features")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Feature importance describes how the fitted Random Forest used the encoded variables in this sample. "
        "It does not establish causation, and importance can be unstable with a small dataset."
    )

# ---------------------------------------------------------
# 8. FINAL CONCLUSION
# ---------------------------------------------------------
st.header("8. Final Conclusion")
st.write(
    "This report organizes the survey into academic performance, academic habits and lifestyle, "
    "and student characteristics. It examines paired sleep-hour differences, an association between "
    "employment and scholarship status, and whether the average previous-semester GPA differs from 8.0. "
    "Finally, Ridge Regression and Random Forest are compared to estimate previous-semester GPA. "
    "The conclusions should be interpreted in light of the sample size, survey responses, and model limitations."
)
