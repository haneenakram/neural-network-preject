import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backend import load_data, preprocessing, choosing_features, filter_classes, train_split, plot_decision_boundary
from SLP import SLP  # Import SLP class
from Adaline import Adaline  # Import Adaline class

# Mapping between numerical classes and display labels
CLASS_MAPPING = {
    0: "A",
    1: "B",
    2: "C"
}

# Fixed class pair combinations
CLASS_PAIRS = {
    "A and B": (0, 1),
    "A and C": (0, 2),
    "B and C": (1, 2)
}

def main():
    st.title("Perceptron Models for Bird Classification")
    
    # Load and preprocess data
    data = load_data("birds.csv")
    data = preprocessing(data)

    # Main panel configuration
    st.header("Model Configuration")
    
    # Model selection
    model_type = st.selectbox("Select Model", ["Single Layer Perceptron (SLP)", "Adaline"])

    # Feature selection
    col1, col2 = st.columns(2)
    with col1:
        feature1 = st.selectbox("Select Feature 1", data.columns[:-1], index=0)
    with col2:
        feature2 = st.selectbox("Select Feature 2", data.columns[:-1], index=1)

    # Class pair selection
    selected_pair_label = st.selectbox("Select Classes to Compare", list(CLASS_PAIRS.keys()))
    class_1, class_2 = CLASS_PAIRS[selected_pair_label]

    # Hyperparameters
    params = st.columns(3)
    with params[0]:
        learning_rate = st.number_input("Learning Rate", 
                                      min_value=0.0, 
                                      max_value=1.0,
                                      value=0.01,
                                      format="%.10f")
    with params[1]:
        epochs = st.number_input("Number of Epochs", 
                               min_value=1, 
                               max_value=10000,
                               value=100,
                               step=10)
    with params[2]:
        use_bias = st.checkbox("Use Bias", value=True)

    # Additional parameter for Adaline
    if model_type == "Adaline":
        mse_threshold = st.number_input("MSE Threshold", 
                                       min_value=0.0, 
                                       max_value=1.0,
                                       value=0.2,
                                       format="%.4f")

    # Filter and split data
    selected_data = choosing_features(data, feature1, feature2)
    filtered_data = filter_classes(selected_data, class_1, class_2)
    selected_features = [feature1, feature2]
    X_train, y_train, X_test, y_test = train_split(selected_features, filtered_data)

    if st.button("Train and Evaluate Model"):
        # Initialize the selected model
        if model_type == "Single Layer Perceptron (SLP)":
            model = SLP(learning_rate=learning_rate, epochs=epochs, Abias=use_bias)
        else:
            model = Adaline(learning_rate=learning_rate, epochs=epochs, Abias=use_bias, MSE_thres=mse_threshold)
        
        # Create a placeholder for training updates
        training_status = st.empty()
        
        # Train the model
        model.train(X_train, y_train)
        
        # Display final training results
        training_status.success("Training completed!")
        
        # Plot training accuracy over epochs
        st.header("Training Progress")
        fig, ax = plt.subplots()
        ax.plot(range(1, len(model.train_accuracy_history) + 1), model.train_accuracy_history, 'b-', label="Training Accuracy")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Training Accuracy Over Epochs")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # For Adaline, also show loss curve
        if model_type == "Adaline":
            st.subheader("Training Loss Over Epochs")
            fig2, ax2 = plt.subplots()
            ax2.plot(range(1, len(model.losses) + 1), model.losses, 'r-', label="Training Loss (MSE)")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Loss (MSE)")
            ax2.set_title("Training Loss Over Epochs")
            ax2.grid(True)
            ax2.legend()
            st.pyplot(fig2)

        # Test the model
        st.header("Evaluation Results")
        confusion_matrix, test_accuracy = model.test(X_test, y_test)
        
        # Display metrics
        metrics = st.columns(2)
        with metrics[0]:
            st.metric("Test Accuracy", f"{test_accuracy:.2f}%")
        with metrics[1]:
            st.write("Confusion Matrix:")
            st.write(confusion_matrix)

        # Decision boundary visualization
        st.header("Decision Boundary")
        fig, ax = plt.subplots()
        plot_decision_boundary(model, X_train, y_train, feature1, feature2)
        st.pyplot(fig)

        # Prediction interface
        st.header("Make Prediction")
        pred_cols = st.columns(2)
        with pred_cols[0]:
            feat1_val = st.number_input(f"{feature1}", value=0.0)
        with pred_cols[1]:
            feat2_val = st.number_input(f"{feature2}", value=0.0)
        
        if st.button("Predict"):
            input_array = np.array([feat1_val, feat2_val])
            prediction = model.predict(input_array)
            class_name = "Class 1" if prediction == 1 else "Class 2"
            st.success(f"Predicted Category: {class_name}")

if __name__ == "__main__":
    main()
