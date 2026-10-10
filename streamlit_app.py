
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Website title
st.title("🌸 Iris Flower Classification")
st.write(
    "Enter the flower measurements to predict "
    "the Iris flower species using KNN."
)

st.divider()

# Display accuracy
st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
st.caption("Machine Learning Algorithm: K-Nearest Neighbors (KNN)")

# User inputs
st.subheader("Enter Flower Measurements")

sepal_length = st.number_input(
    "Sepal Length (cm)", min_value=0.1,
    max_value=10.0, value=5.1, step=0.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)", min_value=0.1,
    max_value=10.0, value=3.5, step=0.1
)

petal_length = st.number_input(
    "Petal Length (cm)", min_value=0.1,
    max_value=10.0, value=1.4, step=0.1
)

petal_width = st.number_input(
    "Petal Width (cm)", min_value=0.1,
    max_value=10.0, value=0.2, step=0.1
)

# Prediction
if st.button("Predict Flower Species", type="primary"):
    new_flower = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(new_flower)
    flower_name = iris.target_names[prediction[0]]

    st.divider()
    st.subheader("Prediction Result")
    st.success(f"Predicted Flower: {flower_name.title()}")
