import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backend import *
from SLP import SLP
from Adaline import Adaline
from sklearn.preprocessing import StandardScaler

CLASS_MAPPING = {0: "A", 1: "B", 2: "C"}
CLASS_PAIRS = {"A and B": (0, 1), "A and C": (0, 2), "B and C": (1, 2)}

def main():
    st.title("Perceptron Models for Bird Classification")
    
    # Initialize session state with class pair information
    if 'trained_model' not in st.session_state:
        st.session_state.update({
            'trained_model': None,
            'feature1': None,
            'feature2': None,
            'scaler': None,
            'numerical_features': [],
            'confusion_matrix': None,
            'test_accuracy': None,
            'X_train': None,
            'y_train': None,
            'class_pair': None,
            'class_names': {}
        })

    # Load and preprocess data
    data = load_data("birds.csv")
    data = preprocessing(data)

    # Model configuration
    st.header("Model Configuration")
    model_type = st.selectbox("Select Model", ["Single Layer Perceptron (SLP)", "Adaline"])
    
    # Feature selection
    col1, col2 = st.columns(2)
    with col1: feature1 = st.selectbox("Select Feature 1", data.columns[:-1], index=0)
    with col2: feature2 = st.selectbox("Select Feature 2", data.columns[:-1], index=1)
    
    # Class pair selection
    selected_pair_label = st.selectbox("Select Classes to Compare", list(CLASS_PAIRS.keys()))
    class_1, class_2 = CLASS_PAIRS[selected_pair_label]

    # Store class names for prediction mapping
    class_names = {
        -1: CLASS_MAPPING[class_1],
        1: CLASS_MAPPING[class_2]
    }

    # Hyperparameters
    learning_rate = st.number_input("Learning Rate", 0.0, 1.0, 0.01, format="%.10f")
    epochs = st.number_input("Number of Epochs", 1, 10000, 100, step=10)
    use_bias = st.checkbox("Use Bias", True)
    mse_threshold = st.number_input("MSE Threshold", 0.0, 1.0, 0.2, format="%.4f") if model_type == "Adaline" else None

    # Data processing
    selected_data = choosing_features(data, feature1, feature2)
    filtered_data = filter_classes(selected_data, class_1, class_2)
    
    if st.button("Train and Evaluate Model"):
        # Split data first
        X_train, y_train, X_test, y_test = train_split([feature1, feature2], filtered_data)
        
        # Scale numerical features properly
        numerical_features = [feat for feat in [feature1, feature2] if feat != 'gender']
        scaler = StandardScaler()
        
        if numerical_features:
            X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
            X_test[numerical_features] = scaler.transform(X_test[numerical_features])
        
        # Initialize and train model
        model = SLP(learning_rate, epochs, use_bias) if model_type.startswith("Single") else Adaline(learning_rate, epochs, use_bias, mse_threshold)
        model.train(X_train, y_train)
        
        # Evaluate model
        confusion_matrix, test_accuracy = model.test(X_test, y_test)
        
        # Store everything in session state
        st.session_state.update({
            'trained_model': model,
            'feature1': feature1,
            'feature2': feature2,
            'scaler': scaler,
            'numerical_features': numerical_features,
            'confusion_matrix': confusion_matrix,
            'test_accuracy': test_accuracy,
            'X_train': X_train,
            'y_train': y_train,
            'class_pair': (class_1, class_2),
            'class_names': class_names
        })

    # Always show results if available
    if st.session_state.confusion_matrix is not None:
        st.success("Training completed!")
        st.header("Training Progress")
        fig, ax = plt.subplots()
        ax.plot(range(1, len(st.session_state.trained_model.train_accuracy_history) + 1),
                st.session_state.trained_model.train_accuracy_history, 'b-', label="Training Accuracy")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        if model_type == "Adaline":
            st.subheader("Training Loss")
            fig2, ax2 = plt.subplots()
            ax2.plot(st.session_state.trained_model.losses, 'r-', label="MSE Loss")
            ax2.set_xlabel("Epochs")
            ax2.grid(True)
            st.pyplot(fig2)

        st.header("Evaluation Results")
        cols = st.columns(2)
        with cols[0]: st.metric("Test Accuracy", f"{st.session_state.test_accuracy:.2f}%")
        with cols[1]: st.write("Confusion Matrix:", st.session_state.confusion_matrix)

        st.header("Decision Boundary")
        fig3, ax3 = plt.subplots()
        plot_decision_boundary(st.session_state.trained_model, 
                              st.session_state.X_train, 
                              st.session_state.y_train, 
                              st.session_state.feature1, 
                              st.session_state.feature2)
        st.pyplot(fig3)

    # Prediction interface
    if st.session_state.trained_model is not None:
        st.header("Make Prediction")
        col1, col2 = st.columns(2)
        input_values = []
        
        with col1:
            if st.session_state.feature1 == 'gender':
                feat1 = st.selectbox(f"{st.session_state.feature1}", ['male', 'female'])
                input_values.append(0 if feat1 == 'male' else 1)
            else:
                feat1 = st.number_input(f"{st.session_state.feature1}", value=0.0)
                input_values.append(float(feat1))
        
        with col2:
            if st.session_state.feature2 == 'gender':
                feat2 = st.selectbox(f"{st.session_state.feature2}", ['male', 'female'])
                input_values.append(0 if feat2 == 'male' else 1)
            else:
                feat2 = st.number_input(f"{st.session_state.feature2}", value=0.0)
                input_values.append(float(feat2))

        if st.button("Predict"):
            input_df = pd.DataFrame([input_values], 
                                  columns=[st.session_state.feature1, st.session_state.feature2])
            
            if st.session_state.numerical_features:
                input_df[st.session_state.numerical_features] = st.session_state.scaler.transform(
                    input_df[st.session_state.numerical_features]
                )
            
            prediction = st.session_state.trained_model.predict(input_df.values)
            predicted_class = st.session_state.class_names.get(prediction[0], "Unknown")
            st.success(f"Predicted Category: {predicted_class}")

if __name__ == "__main__":
    main()