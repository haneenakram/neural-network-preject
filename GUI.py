# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backend import load_data, preprocessing, choosing_features, filter_classes, train_split, plot_decision_boundary
from SLP import SLP

def main():
    st.title("Single Layer Perceptron (SLP) for Bird Classification")
    
    # Load and preprocess data
    data = load_data("birds.csv")
    data = preprocessing(data)

    # Main panel configuration
    st.header("Model Configuration")
    
    # Feature selection in main panel
    col1, col2 = st.columns(2)
    with col1:
        feature1 = st.selectbox("Select Feature 1", data.columns[:-1], index=0)
    with col2:
        feature2 = st.selectbox("Select Feature 2", data.columns[:-1], index=1)

    # Class selection
    class1, class2 = st.columns(2)
    with class1:
        class_1 = st.selectbox("Select First Class", data['bird category'].unique(), index=0)
    with class2:
        class_2 = st.selectbox("Select Second Class", data['bird category'].unique(), index=1)

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

    # Filter and split data
    selected_data = choosing_features(data, feature1, feature2)
    filtered_data = filter_classes(selected_data, class_1, class_2)
    selected_features = [feature1, feature2]
    X_train, y_train, X_test, y_test = train_split(selected_features, filtered_data)

    if st.button("Train and Evaluate Model"):
        # Initialize and train model
        slp = SLP(learning_rate=learning_rate, epochs=epochs, Abias=use_bias)
        
        # Create a placeholder for training updates
        training_status = st.empty()
        
        # Train the model
        slp.train(X_train, y_train)
        
        # Display final training results
        training_status.success("Training completed!")
        
        # Plot training progress
        st.header("Training Progress")
        fig, ax = plt.subplots()
        ax.plot(range(1, epochs+1), slp.train_accuracy_history, 'b-')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Training Accuracy (%)")
        ax.set_title("Training Accuracy Progress")
        ax.grid(True)
        st.pyplot(fig)

        # Test the model
        st.header("Evaluation Results")
        confusion_matrix, test_accuracy = slp.test(X_test, y_test)
        
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
        plot_decision_boundary(slp, X_train, y_train, feature1, feature2)
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
            prediction = slp.predict(input_array)
            class_name = "Class 1" if prediction == 1 else "Class 2"
            st.success(f"Predicted Category: {class_name}")

if __name__ == "__main__":
    main()