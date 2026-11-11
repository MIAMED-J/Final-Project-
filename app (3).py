import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st

from sklearn.compose import ColumnTransformer
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
    "This report analyzes 118 UAP survey responses through descriptive statistics, "
    "hypothesis testing, and grade prediction models."
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
    st.error("The file 'Midterm Grade Survey.csv' was not found. Save it in the same folder as app.py.")
    st.stop()

st.success(f"Dataset loaded: {len(datos)} survey responses and {len(datos.columns)} original survey variables.")

# Exact column names from the survey
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

required_columns = [student_type, school, subjects, study_hours, study_method, midterm_grade,
    gpa_column, scholarship, study_start, routine_affected, regular_sleep, midterm_sleep,
    stress, self_rated, employment, extracurricular, financial_support, income]
missing_columns = [col for col in required_columns if col not in datos.columns]
if missing_columns:
    st.error("Some survey columns were not found. Make sure you uploaded the updated CSV file.")
    st.write(missing_columns)
    st.stop()

# ---------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------
def number_from_answer(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace(",", ".")
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if not numbers:
        return np.nan
    values = [float(number) for number in numbers]
    if len(values) >= 2 and ("-" in text or " a " in text or " to " in text):
        return float(np.mean(values[:2]))
    return values[0]


def income_to_number(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return float(numbers[0]) if numbers else np.nan


def clean_category(series):
    return series.astype("string").str.strip().replace("", pd.NA)

# GPA values above 10 are interpreted as values entered without a decimal point
# (for example, 92 becomes 9.2). Non-numeric and out-of-range answers are missing.
datos["GPA_clean"] = pd.to_numeric(datos[gpa_column].astype(str).str.strip(), errors="coerce")
datos.loc[datos["GPA_clean"] > 10, "GPA_clean"] = datos.loc[datos["GPA_clean"] > 10, "GPA_clean"] / 10
datos.loc[~datos["GPA_clean"].between(0, 10), "GPA_clean"] = np.nan

datos["Study_hours_clean"] = datos[study_hours].apply(number_from_answer)
datos["Income_clean"] = datos[income].apply(income_to_number)
for col in [subjects, midterm_grade, routine_affected, regular_sleep, midterm_sleep, stress, self_rated]:
    datos[col + "_numeric"] = datos[col].apply(number_from_answer)

for col in [student_type, school, study_method, scholarship, study_start,
            employment, extracurricular, financial_support]:
    datos[col] = clean_category(datos[col])

# Short aliases for analysis
subjects_num = subjects + "_numeric"
midterm_grade_num = midterm_grade + "_numeric"
routine_num = routine_affected + "_numeric"
regular_sleep_num = regular_sleep + "_numeric"
midterm_sleep_num = midterm_sleep + "_numeric"
stress_num = stress + "_numeric"
self_rated_num = self_rated + "_numeric"

alpha = 0.05

# ---------------------------------------------------------
# 1. VARIABLE GROUPS AND RESEARCH APPROACH
# ---------------------------------------------------------
st.header("1. Variable Groups and Research Approach")
st.write(
    "The survey variables are classified into three conceptual groups according to the type of "
    "information they represent. This organization helps structure the descriptive analysis, "
    "research questions, and prediction models. The groups are a way to organize the survey; "
    "they do not by themselves prove that a variable is important or causes academic performance."
)

with st.expander("Group 1 – Academic Performance", expanded=True):
    st.write("**Purpose:** Describe students' academic results.")
    st.markdown("- Previous-semester GPA (**main prediction target**)\n- Average grade on recent midterm exams\n- Self-rated midterm performance")
with st.expander("Group 2 – Academic Habits and Lifestyle", expanded=True):
    st.write("**Purpose:** Explore study routines and lifestyle factors that may be associated with academic results.")
    st.markdown("- Number of subjects\n- Weekly study hours\n- Study method and study start time\n- Sleep hours during a regular week and midterm week\n- Stress level\n- How much the routine is affected by midterms")
with st.expander("Group 3 – Student Characteristics", expanded=True):
    st.write("**Purpose:** Describe students' academic, employment, and financial context.")
    st.markdown("- Student type and academic school\n- Scholarship status\n- Employment status\n- Extracurricular activities\n- Main financial support\n- Monthly income")

st.info(
    "The previous-semester GPA is the target for the prediction models. Current midterm grades "
    "and self-rated midterm performance are described separately and are not used as predictors, "
    "because they refer to a different/current academic period."
)

# ---------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------
st.header("2. Descriptive Statistics")
st.write(
    "Descriptive statistics summarize the responses before hypothesis testing. "
    "For numeric variables, the report shows the number of valid answers, mean, median, "
    "standard deviation, minimum, and maximum. For categorical variables, it shows response counts."
)
metric1, metric2, metric3 = st.columns(3)
metric1.metric("Survey responses", len(datos))
metric2.metric("Valid previous-semester GPA", int(datos["GPA_clean"].notna().sum()))
metric3.metric("Significance level", "0.05")

st.subheader("Group 1 – Academic Performance")
performance_data = datos[["GPA_clean", midterm_grade_num, self_rated_num]].copy()
performance_data.columns = ["Previous-semester GPA", "Average midterm grade", "Self-rated midterm performance"]
st.dataframe(performance_data.describe().T.round(2), use_container_width=True)
st.caption("This table is a summary of the academic performance measures. It is descriptive, not a hypothesis test.")

st.subheader("Group 2 – Academic Habits and Lifestyle")
habits_data = datos[[subjects_num, "Study_hours_clean", regular_sleep_num, midterm_sleep_num, stress_num, routine_num]].copy()
habits_data.columns = ["Number of subjects", "Study hours per week", "Regular-week sleep hours", "Midterm-week sleep hours", "Stress level", "Routine affected"]
st.dataframe(habits_data.describe().T.round(2), use_container_width=True)
st.caption("This table summarizes numeric study and lifestyle variables. Study method and study start time are categorical, so they are summarized with counts below.")

with st.expander("Categorical response counts", expanded=False):
    for label, col in [("Study method", study_method), ("Study start time", study_start),
                       ("Student type", student_type), ("School", school),
                       ("Scholarship", scholarship), ("Employment", employment),
                       ("Extracurricular activities", extracurricular),
                       ("Financial support", financial_support)]:
        st.write(f"**{label}**")
        st.dataframe(datos[col].value_counts(dropna=False).rename("Number of students").to_frame(), use_container_width=True)

# ---------------------------------------------------------
# 3. DATA VISUALIZATION
# ---------------------------------------------------------
st.header("3. Data Visualization")
plot_col1, plot_col2 = st.columns(2)
with plot_col1:
    sleep_means = [datos[regular_sleep_num].mean(), datos[midterm_sleep_num].mean()]
    fig, ax = plt.subplots()
    ax.bar(["Regular week", "Midterm week"], sleep_means)
    ax.set_ylabel("Average hours of sleep")
    ax.set_title("Average Sleep Hours")
    st.pyplot(fig)
    plt.close(fig)
with plot_col2:
    fig, ax = plt.subplots()
    datos["GPA_clean"].dropna().hist(ax=ax, bins=10)
    ax.set_xlabel("Previous-semester GPA")
    ax.set_ylabel("Number of students")
    ax.set_title("Previous-Semester GPA Distribution")
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------
# 4. SPEARMAN CORRELATION – NUMERICAL VARIABLES
# ---------------------------------------------------------
st.header("4. Spearman Correlation: Relationships with Midterm Grades")
st.write(
    "Spearman correlation is used to examine monotonic relationships between numerical or ordinal variables. "
    "It does not require the variables to follow a normal distribution and does not establish causation."
)
st.write(
    "The main outcome for this section is the average grade on recent midterm exams (Group 1). "
    "We examine whether habits, lifestyle, and student characteristics represented by numeric or ordinal "
    "variables are associated with current midterm grades. We also examine selected relationships within Group 2."
)

spearman_pairs = [
    ("Weekly study hours", "Study_hours_clean", "Average midterm grade", midterm_grade_num),
    ("Regular-week sleep hours", regular_sleep_num, "Average midterm grade", midterm_grade_num),
    ("Midterm-week sleep hours", midterm_sleep_num, "Average midterm grade", midterm_grade_num),
    ("Stress level", stress_num, "Average midterm grade", midterm_grade_num),
    ("Number of subjects", subjects_num, "Average midterm grade", midterm_grade_num),
    ("Routine affected by midterms", routine_num, "Average midterm grade", midterm_grade_num),
    ("Monthly income", "Income_clean", "Average midterm grade", midterm_grade_num),
    ("Previous-semester GPA", "GPA_clean", "Average midterm grade", midterm_grade_num),
    ("Self-rated midterm performance", self_rated_num, "Average midterm grade", midterm_grade_num),
    ("Weekly study hours", "Study_hours_clean", "Stress level", stress_num),
    ("Regular-week sleep hours", regular_sleep_num, "Stress level", stress_num),
    ("Weekly study hours", "Study_hours_clean", "Regular-week sleep hours", regular_sleep_num),
    ("Routine affected by midterms", routine_num, "Stress level", stress_num),
]

spearman_results = []
for label_x, col_x, label_y, col_y in spearman_pairs:
    pair = datos[[col_x, col_y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) >= 3 and pair[col_x].nunique() > 1 and pair[col_y].nunique() > 1:
        rho, p_corr = stats.spearmanr(pair[col_x], pair[col_y])
        spearman_results.append({
            "Variable 1": label_x,
            "Variable 2": label_y,
            "Valid responses": len(pair),
            "Spearman rho": rho,
            "P-value": p_corr,
            "Statistically significant": "Yes" if p_corr < alpha else "No",
        })
    else:
        spearman_results.append({
            "Variable 1": label_x,
            "Variable 2": label_y,
            "Valid responses": len(pair),
            "Spearman rho": np.nan,
            "P-value": np.nan,
            "Statistically significant": "Not enough data/variation",
        })

if spearman_results:
    spearman_table = pd.DataFrame(spearman_results)
    st.dataframe(spearman_table.round(4), use_container_width=True)
    st.caption("H0 for each pair: Spearman rho = 0 (no monotonic association). H1: Spearman rho ≠ 0 (a monotonic association exists). The primary outcome is the average grade on recent midterm exams.")
    st.write("A p-value below 0.05 provides evidence of an association. The sign of rho indicates its direction; its absolute value indicates its strength.")

# ---------------------------------------------------------
# 5. GROUP 2 – PAIRED SLEEP TESTS
# ---------------------------------------------------------
st.header("5. Group 2 – Sleep Hours: Paired Non-Parametric Tests")
st.write(
    "The same students reported their sleep during a regular academic week and during midterm week. "
    "Because these are two measurements from each student, the observations are paired."
)
st.subheader("5.1 Sign Test")
st.write("**Research question:** Is one period more likely to have higher sleep hours than the other?")
st.write("**H0:** Positive and negative paired differences are equally likely (median difference is zero).")
st.write("**H1:** Positive and negative paired differences are not equally likely (median difference is not zero).")
sleep_data = datos[[regular_sleep_num, midterm_sleep_num]].apply(pd.to_numeric, errors="coerce").dropna()
x = sleep_data[regular_sleep_num]
y_sleep = sleep_data[midterm_sleep_num]
differences = x - y_sleep
nonzero_differences = differences[differences != 0]
positive = int((nonzero_differences > 0).sum())
negative = int((nonzero_differences < 0).sum())
ties = int((differences == 0).sum())
st.write(f"Students with both answers: {len(sleep_data)}")
st.write(f"Regular-week mean: {x.mean():.2f} hours | Midterm-week mean: {y_sleep.mean():.2f} hours")
st.write(f"Regular week > midterm week: {positive} | Regular week < midterm week: {negative} | Ties: {ties}")
if len(nonzero_differences) > 0:
    sign_result = stats.binomtest(positive, n=len(nonzero_differences), p=0.5, alternative="two-sided")
    p_sign = sign_result.pvalue
    st.write(f"Sign Test p-value: {p_sign:.4f}")
    if p_sign < alpha:
        st.success("Decision: Reject H0. The direction of the paired sleep differences is not balanced.")
    else:
        st.info("Decision: Do not reject H0. There is not enough evidence that one period tends to have higher sleep hours.")
else:
    st.warning("The Sign Test cannot be calculated because all paired differences are zero.")

st.subheader("5.2 Wilcoxon Signed-Rank Test")
st.write("**Research question:** Is the paired sleep-hour difference between regular week and midterm week centered around zero?")
st.write("**H0:** The distribution of paired differences is centered at zero.")
st.write("**H1:** The distribution of paired differences is not centered at zero.")
if len(sleep_data) >= 2 and len(nonzero_differences) > 0:
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(x, y_sleep, alternative="two-sided")
        st.write(f"Wilcoxon signed-rank statistic: {w_stat:.4f} | P-value: {p_wilcoxon:.4f}")
        if p_wilcoxon < alpha:
            st.success("Decision: Reject H0. There is evidence of a difference in paired sleep hours between the two periods.")
        else:
            st.info("Decision: Do not reject H0. There is not enough evidence of a difference in paired sleep hours.")
    except ValueError as error:
        st.warning(f"Wilcoxon signed-rank test could not be calculated: {error}")
else:
    st.warning("There are not enough non-zero paired differences to perform the Wilcoxon signed-rank test.")

st.caption(
    "The Sign Test uses only the direction of each paired difference. The Wilcoxon signed-rank test "
    "also considers the ranks of the absolute differences. Both tests compare sleep across the two periods; "
    "they do not test whether sleep causes or determines GPA."
)

# ---------------------------------------------------------
# 6. GROUP 3 AND CROSS-GROUP CATEGORICAL ASSOCIATIONS
# ---------------------------------------------------------
st.header("6. Chi-Square Tests of Independence")
st.write(
    "Chi-square tests examine whether two categorical variables are statistically independent "
    "or associated. These tests can be used within a group or between groups, depending on the research question."
)
chi_square_pairs = [
    ("Employment status", employment, "Scholarship status", scholarship),
    ("Student type", student_type, "Scholarship status", scholarship),
    ("Academic school", school, "Scholarship status", scholarship),
    ("Employment status", employment, "Extracurricular activities", extracurricular),
    ("Study method", study_method, "Study start time", study_start),
]
chi_results = []
for label_a, col_a, label_b, col_b in chi_square_pairs:
    chi_data = datos[[col_a, col_b]].copy()
    chi_data[col_a] = clean_category(chi_data[col_a])
    chi_data[col_b] = clean_category(chi_data[col_b])
    chi_data = chi_data.dropna()
    table = pd.crosstab(chi_data[col_a], chi_data[col_b])
    with st.expander(f"{label_a} vs. {label_b}"):
        st.write(f"**H0:** {label_a} and {label_b} are independent.")
        st.write(f"**H1:** {label_a} and {label_b} are associated.")
        st.write("Observed frequency table:")
        st.dataframe(table, use_container_width=True)
        if table.shape[0] >= 2 and table.shape[1] >= 2:
            chi_stat, chi_p, chi_df, expected = stats.chi2_contingency(table, correction=False)
            st.write(f"Chi-square statistic: {chi_stat:.4f} | Degrees of freedom: {chi_df} | P-value: {chi_p:.4f}")
            st.write(f"Minimum expected frequency: {expected.min():.2f}")
            if (expected < 5).any():
                st.warning("Some expected frequencies are below 5. Interpret this chi-square result cautiously; categories may need to be combined when substantively justified.")
            if chi_p < alpha:
                decision = "Yes"
                st.success("Decision: Reject H0. There is evidence of an association between these categorical variables.")
            else:
                decision = "No"
                st.info("Decision: Do not reject H0. There is not enough evidence of an association between these categorical variables.")
            chi_results.append({"Variable 1": label_a, "Variable 2": label_b, "Valid responses": len(chi_data), "P-value": chi_p, "Significant": decision})
        else:
            st.warning("A chi-square test requires at least two observed categories in each variable.")
            chi_results.append({"Variable 1": label_a, "Variable 2": label_b, "Valid responses": len(chi_data), "P-value": np.nan, "Significant": "Not testable"})

if chi_results:
    st.subheader("Chi-Square Summary")
    st.dataframe(pd.DataFrame(chi_results).round(4), use_container_width=True)

st.caption("A significant chi-square result indicates association, not causation. Check expected cell counts before interpreting the p-value.")

# ---------------------------------------------------------
# 7. MACHINE LEARNING – RIDGE REGRESSION
# ---------------------------------------------------------
st.header("7. Grade Prediction Model – Ridge Regression")
st.write(
    "Ridge Regression estimates previous-semester GPA using selected variables from Group 2 "
    "(academic habits and lifestyle) and Group 3 (student characteristics). "
    "Ridge adds a penalty to large coefficients to help reduce overfitting when predictors are correlated."
)
st.info(
    "This is a retrospective prediction exercise: the survey was collected during the current midterm period, "
    "while the target is previous-semester GPA. The model evaluates whether current reported habits and "
    "characteristics contain information associated with past GPA. It does not prove causation or guarantee future GPA predictions."
)

numeric_features = [subjects_num, "Study_hours_clean", routine_num, regular_sleep_num, midterm_sleep_num, stress_num, "Income_clean"]
categorical_features = [student_type, school, study_method, scholarship, study_start, employment, extracurricular, financial_support]
feature_columns = numeric_features + categorical_features
model_data = datos[feature_columns + ["GPA_clean"]].copy().dropna(subset=["GPA_clean"])
X = model_data[feature_columns].copy()
y_model = model_data["GPA_clean"].copy()
for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")
for col in categorical_features:
    X[col] = X[col].astype("string").fillna("Unknown").astype(str)

st.write(f"Students with valid previous-semester GPA available for modeling: {len(model_data)}")
if len(model_data) < 20:
    st.warning("There are too few valid GPA observations to train and evaluate the model reliably.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y_model, test_size=0.20, random_state=42)
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ])
    ridge_model = Pipeline(steps=[("preprocessor", preprocessor), ("model", Ridge(alpha=1.0))])
    ridge_model.fit(X_train, y_train)
    ridge_predictions = ridge_model.predict(X_test)

    ridge_metrics = {
        "MAE": mean_absolute_error(y_test, ridge_predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, ridge_predictions)),
        "R-squared": r2_score(y_test, ridge_predictions),
    }
    st.subheader("7.1 Ridge Regression Evaluation")
    st.write("The model is evaluated on a 20% test sample that was not used during training.")
    metric_table = pd.DataFrame([ridge_metrics], index=["Ridge Regression"]).round(3)
    st.dataframe(metric_table, use_container_width=True)
    st.write("**MAE:** average absolute prediction error in GPA points. Lower is better.")
    st.write("**RMSE:** gives greater weight to larger prediction errors. Lower is better.")
    st.write("**R-squared:** proportion of test-sample GPA variation explained by the model. Values closer to 1 indicate better fit; a negative value means the model performs worse than predicting the test-set mean.")

    st.subheader("7.2 Variables Most Associated with the Ridge Prediction")
    fitted_preprocessor = ridge_model.named_steps["preprocessor"]
    encoded_names = fitted_preprocessor.get_feature_names_out()
    coefficients = ridge_model.named_steps["model"].coef_
    original_coefficients = {}
    for encoded_name, coefficient in zip(encoded_names, coefficients):
        original_name = None
        for feature in feature_columns:
            if encoded_name.startswith("numeric__" + feature) or encoded_name.startswith("categorical__" + feature + "_"):
                original_name = feature
                break
        if original_name is not None:
            original_coefficients[original_name] = original_coefficients.get(original_name, 0) + abs(float(coefficient))
    friendly_names = {
        subjects_num: "Number of subjects",
        "Study_hours_clean": "Weekly study hours",
        routine_num: "Routine affected by midterms",
        regular_sleep_num: "Regular-week sleep hours",
        midterm_sleep_num: "Midterm-week sleep hours",
        stress_num: "Stress level",
        "Income_clean": "Monthly income",
        student_type: "Student type",
        school: "Academic school",
        study_method: "Study method",
        scholarship: "Scholarship status",
        study_start: "Study start time",
        employment: "Employment status",
        extracurricular: "Extracurricular activities",
        financial_support: "Financial support",
    }
    coefficient_series = pd.Series(original_coefficients).sort_values(ascending=False).head(8)
    coefficient_display = coefficient_series.rename(index=friendly_names)
    fig, ax = plt.subplots(figsize=(9, 5))
    coefficient_display.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Sum of absolute Ridge coefficients (encoded categories combined)")
    ax.set_ylabel("Original survey variable")
    ax.set_title("Ridge Regression: Largest Coefficient Magnitudes")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Coefficient magnitude indicates model contribution on the transformed scale, not causation. Categorical coefficients depend on the reference category and encoding.")

    st.subheader("7.3 Model Interpretation")
    if ridge_metrics["R-squared"] >= 0:
        st.write(f"On the test sample, Ridge Regression explained approximately {ridge_metrics['R-squared'] * 100:.1f}% of the variation in previous-semester GPA.")
    else:
        st.write("The negative R-squared indicates that the model did not outperform a simple prediction based on the average GPA in this test sample.")
    st.caption("The dataset is relatively small, so results may change with a different train/test split. Interpret the metrics as an initial evaluation, not as a definitive measure of predictive performance.")

# ---------------------------------------------------------
# 8. FINAL CONCLUSION
# ---------------------------------------------------------
st.header("8. Final Conclusion")
st.write(
    "The analysis is organized around three groups: academic performance, academic habits and lifestyle, "
    "and student characteristics. Spearman correlation examines monotonic relationships between numerical "
    "variables, mainly in relation to recent midterm grades; the Sign Test and Wilcoxon signed-rank test compare paired sleep reports from regular and "
    "midterm weeks; and Chi-Square tests examine independence between selected categorical variables. "
    "Ridge Regression evaluates how well habits and student characteristics estimate previous-semester GPA."
)
st.write(
    "A statistically significant association does not establish cause and effect. A non-significant result "
    "does not prove that no relationship exists. Conclusions must consider the sample size, missing responses, "
    "expected frequencies, and the retrospective nature of the GPA prediction."
)
