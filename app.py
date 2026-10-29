import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# TITLE
st.title("UAP Students Statistics Report")
st.write("Descriptive Statistics and Hypothesis Testing")

# READ THE DATA
datos = pd.read_csv("Midterm Grade Survey.csv")

# SELECT THE TWO SURVEY QUESTIONS
regular_sleep = "On average, how many hours of sleep do you get per night during a regular academic week?\n"
midterm_sleep = "On average, how many hours of sleep did you get on midterm week?"

x = datos[regular_sleep].dropna()
y = datos[midterm_sleep].dropna()

# USE STUDENTS WITH ANSWERS TO BOTH QUESTIONS
sleep = datos[[regular_sleep, midterm_sleep]].dropna()

x = sleep[regular_sleep]
y = sleep[midterm_sleep]

# 1. DESCRIPTIVE STATISTICS
st.header("1. Descriptive Statistics")

n = len(sleep)

st.write("Number of students:", n)

st.subheader("Regular Academic Week")
st.write("Mean:", x.mean())
st.write("Median:", x.median())
st.write("Standard deviation:", x.std())
st.write("Minimum:", x.min())
st.write("Maximum:", x.max())

st.subheader("Midterm Week")
st.write("Mean:", y.mean())
st.write("Median:", y.median())
st.write("Standard deviation:", y.std())
st.write("Minimum:", y.min())
st.write("Maximum:", y.max())

# 2. VISUALIZATION
st.header("2. Data Visualization")

fig, ax = plt.subplots()

ax.bar(
    ["Regular Week", "Midterm Week"],
    [x.mean(), y.mean()]
)

ax.set_ylabel("Average Hours of Sleep")
ax.set_title("Average Sleep Hours Comparison")

st.pyplot(fig)
plt.close(fig)

# 3. PARAMETRIC TEST: PAIRED T-TEST
st.header("3. Parametric Hypothesis Test")
st.subheader("Paired T-Test")

# H0: The mean difference is zero
# H1: The mean difference is not zero

alpha = 0.05

difference = x - y
sample_mean = difference.mean()
sigma = difference.std()

t_stat, p_value = stats.ttest_rel(x, y)

st.write("H0: The mean difference in sleep hours is zero.")
st.write("H1: The mean difference in sleep hours is not zero.")
st.write("Significance level:", alpha)
st.write("Mean difference:", sample_mean)
st.write("Standard deviation of differences:", sigma)
st.write("T statistic:", t_stat)
st.write("P-value:", p_value)

if p_value < alpha:
    st.write("Decision: Reject H0.")
    st.write("Conclusion: There is a significant difference in sleep hours between regular academic week and midterm week.")
else:
    st.write("Decision: Do not reject H0.")
    st.write("Conclusion: There is not enough evidence of a difference in sleep hours.")

# 4. NON-PARAMETRIC TEST: WILCOXON
st.header("4. Non-Parametric Hypothesis Test")
st.subheader("Wilcoxon Signed-Rank Test")

# H0: The paired differences are centered at zero
# H1: The paired differences are not centered at zero

w_stat, p_wilcoxon = stats.wilcoxon(x, y)

st.write("H0: The paired differences are centered at zero.")
st.write("H1: The paired differences are not centered at zero.")
st.write("Significance level:", alpha)
st.write("Wilcoxon statistic:", w_stat)
st.write("P-value:", p_wilcoxon)

if p_wilcoxon < alpha:
    st.write("Decision: Reject H0.")
    st.write("Conclusion: There is a significant difference in paired sleep hours.")
else:
    st.write("Decision: Do not reject H0.")
    st.write("Conclusion: There is not enough evidence of a difference in paired sleep hours.")

# 5. FINAL CONCLUSION
st.header("5. Final Conclusion")

st.write("Average sleep during a regular academic week:", x.mean())
st.write("Average sleep during midterm week:", y.mean())

if p_value < alpha and p_wilcoxon < alpha:
    st.write("Both tests reject H0. The results provide evidence that students sleep different amounts during a regular academic week and midterm week.")
else:
    st.write("At least one test does not reject H0. Interpret the results considering the assumptions of each test.")
