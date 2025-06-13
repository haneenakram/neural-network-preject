import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from MLP import MLP
from src.utils import read_file, data_encode, standardize_data, filter_classes, train_test_split_data

# Set page configuration
st.set_page_config(page_title="MLP Neural Network Classifier", layout="wide")

# Title
st.title("Multi-Layer Perceptron (MLP) Neural Network for Bird Classification")

# Sidebar for inputs
st.sidebar.header("Model Configuration")

# Input parameters
st.sidebar.subheader("Network Architecture")
num_hidden_layers = st.sidebar.slider("Number of Hidden Layers", min_value=1, max_value=5, value=2)

# Collect neuron counts for each layer
neurons_per_layer = []
for i in range(num_hidden_layers):
    neurons = st.sidebar.number_input(f"Neurons in Hidden Layer {i+1}", min_value=1, max_value=50, value=3)
    neurons_per_layer.append(neurons)

# Training parameters
st.sidebar.subheader("Training Parameters")
learning_rate = st.sidebar.slider("Learning Rate (eta)", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f")
epochs = st.sidebar.slider("Number of Epochs", min_value=100, max_value=5000, value=1000, step=100)
add_bias = st.sidebar.checkbox("Add Bias", value=True)
activation_function = st.sidebar.selectbox("Activation Function", ["sigmoid", "tanh"])

# Define function to load and process data
@st.cache_data
def load_bird_data(file_path="birds.csv"):
    data = pd.read_csv(file_path)
    data_encode(data)
    data = standardize_data(data)
    filtered_data = filter_classes(data, 0, 1, 2)
    return filtered_data

# Main content
st.header("Bird Classification")

# Define tabs
tab1, tab2, tab3 = st.tabs(["Train & Test", "Model Performance", "Single Sample Classification"])

with tab1:
    st.subheader("Training and Testing")
    
    try:
        # Load and preprocess data
        data = load_bird_data()
        
        # Display raw data sample
        st.write("Data Sample (First 5 rows):")
        st.dataframe(data.head())
        
        # Split data: first 30 samples for training, remaining 20 for testing
        X = data.drop(columns=['bird category']).values
        y = data['bird category'].values
        
        # Get first 30 samples per class for training
        train_indices = []
        test_indices = []
        
        for class_label in np.unique(y):
            class_indices = np.where(y == class_label)[0]
            train_indices.extend(class_indices[:30])
            test_indices.extend(class_indices[30:50])
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        
        st.write(f"Training samples: {len(X_train)}")
        st.write(f"Testing samples: {len(X_test)}")
        
        # Train button
        if st.button("Train Model"):
            st.write("Training the neural network...")
            
            # Create and train MLP model
            with st.spinner('Training in progress...'):
                # Initialize MLP
                mlp = MLP(
                    inputSize=X_train.shape[1],
                    hiddenLayers=num_hidden_layers,
                    neurons=neurons_per_layer,
                    outputSize=3,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    Abias=add_bias,
                    activationFn=activation_function
                )
                
                # Capture training progress in a placeholder
                progress_placeholder = st.empty()
                
                # Original train function logs to console, let's capture and show in Streamlit
                import sys
                from io import StringIO
                
                # Redirect stdout to capture print statements
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                
                # Train the model
                mlp.train(X_train, y_train)
                
                # Restore stdout
                sys.stdout = old_stdout
                training_log = mystdout.getvalue()
                
                # Display training log
                progress_placeholder.text_area("Training Progress", training_log, height=300)
                
                # Save model and data to session state
                st.session_state['mlp_model'] = mlp
                st.session_state['X_train'] = X_train
                st.session_state['y_train'] = y_train
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                
                st.success("Training completed!")
                st.session_state['model_trained'] = True
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        st.info("Make sure the 'birds.csv' file is in the same directory as this script.")

with tab2:
    st.subheader("Model Performance")
    
    if 'model_trained' in st.session_state and st.session_state['model_trained']:
        # Get model and data from session state
        mlp = st.session_state['mlp_model']
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        # Create a two-column layout for training and test performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Training Data Performance")
            
            # Make predictions on training data
            y_train_pred_probs = mlp.predict(X_train)
            y_train_pred = np.argmax(y_train_pred_probs, axis=1)
            
            # Calculate training accuracy
            train_accuracy = accuracy_score(y_train, y_train_pred)
            st.write(f"**Training Accuracy:** {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
            
            # Display training confusion matrix
            cm_train = confusion_matrix(y_train, y_train_pred)
            fig_train_cm, ax_train_cm = plt.subplots(figsize=(4, 3))
            disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
            disp_train.plot(cmap=plt.cm.Blues, ax=ax_train_cm)
            plt.title('Training Confusion Matrix')
            plt.tight_layout()
            st.pyplot(fig_train_cm, use_container_width=False)
            
            # Display training classification report
            st.write("#### Classification Report")
            train_report = classification_report(y_train, y_train_pred, output_dict=True)
            train_report_df = pd.DataFrame(train_report).transpose()
            st.table(train_report_df)
            
            # Plot training classification report as heatmap
            fig_train_report, ax_train_report = plt.subplots(figsize=(5, 3))
            sns.heatmap(train_report_df.iloc[:-1, :3].astype(float), annot=True, fmt=".2f", cmap="Blues", ax=ax_train_report)
            plt.title('Training Classification Report Visualization')
            st.pyplot(fig_train_report, use_container_width=False)
        
        with col2:
            st.write("### Test Data Performance")
            
            # Make predictions on test data
            y_test_pred_probs = mlp.predict(X_test)
            y_test_pred = np.argmax(y_test_pred_probs, axis=1)
            
            # Calculate test accuracy
            test_accuracy = accuracy_score(y_test, y_test_pred)
            st.write(f"**Test Accuracy:** {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            
            # Display test confusion matrix
            cm_test = confusion_matrix(y_test, y_test_pred)
            fig_test_cm, ax_test_cm = plt.subplots(figsize=(4, 3))
            disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
            disp_test.plot(cmap=plt.cm.Blues, ax=ax_test_cm)
            plt.title('Test Confusion Matrix')
            plt.tight_layout()
            st.pyplot(fig_test_cm, use_container_width=False)
            
            # Display test classification report
            st.write("#### Classification Report")
            test_report = classification_report(y_test, y_test_pred, output_dict=True)
            test_report_df = pd.DataFrame(test_report).transpose()
            st.table(test_report_df)
            
            # Plot test classification report as heatmap
            fig_test_report, ax_test_report = plt.subplots(figsize=(5, 3))
            sns.heatmap(test_report_df.iloc[:-1, :3].astype(float), annot=True, fmt=".2f", cmap="Blues", ax=ax_test_report)
            plt.title('Test Classification Report Visualization')
            st.pyplot(fig_test_report, use_container_width=False)
        
        # Compare training vs test performance (bar chart)
        st.write("### Training vs Test Accuracy Comparison")
        comparison_df = pd.DataFrame({
            'Dataset': ['Training', 'Testing'],
            'Accuracy': [train_accuracy, test_accuracy]
        })
        
        fig_comparison, ax_comparison = plt.subplots(figsize=(6, 3))
        sns.barplot(x='Dataset', y='Accuracy', data=comparison_df, ax=ax_comparison)
        plt.title('Training vs Test Accuracy')
        plt.ylim(0, 1)
        st.pyplot(fig_comparison, use_container_width=False)
        
        # Summary of parameters used
        st.write("### Model Parameters")
        params_df = pd.DataFrame({
            'Parameter': ['Hidden Layers', 'Neurons per Layer', 'Learning Rate', 'Epochs', 'Bias', 'Activation Function'],
            'Value': [
                num_hidden_layers,
                str(neurons_per_layer),
                learning_rate,
                epochs,
                'Yes' if add_bias else 'No',
                activation_function
            ]
        })
        st.table(params_df)
    else:
        st.info("Train the model first to see performance metrics.")

with tab3:
    st.subheader("Single Sample Classification")
    
    if 'model_trained' in st.session_state and st.session_state['model_trained']:
        st.write("Enter values for a single sample to classify:")
        
        # Create 5 input fields for features
        features = []
        col1, col2 = st.columns(2)
        
        with col1:
            for i in range(3):
                feature = st.number_input(f"Feature {i+1}", value=0.0, format="%.4f")
                features.append(feature)
        
        with col2:
            for i in range(3, 5):
                feature = st.number_input(f"Feature {i+1}", value=0.0, format="%.4f")
                features.append(feature)
        
        if st.button("Classify Sample"):
            # Get the model
            mlp = st.session_state['mlp_model']
            
            # Convert input to numpy array
            sample = np.array([features])
            
            # Make prediction
            prediction_probs = mlp.predict(sample)
            predicted_class = np.argmax(prediction_probs)
            
            # Map class index to bird category
            bird_categories = {0: 'A', 1: 'B', 2: 'C'}
            predicted_bird = bird_categories[predicted_class]
            
            # Show prediction
            st.write(f"### Prediction Results")
            st.write(f"Predicted Bird Category: **{predicted_bird}** (Class {predicted_class})")
            
            # Display probabilities
            probs_df = pd.DataFrame({
                'Bird Category': list(bird_categories.values()),
                'Probability': prediction_probs[0]
            })
            
            # Plot probability distribution
            fig, ax = plt.subplots(figsize=(4, 2))
            sns.barplot(x='Bird Category', y='Probability', data=probs_df, ax=ax)
            plt.title('Classification Probabilities')
            plt.ylim(0, 1)
            st.pyplot(fig,use_container_width=False)
    else:
        st.info("Train the model first to classify samples.")

# Add some information about the app
st.sidebar.markdown("---")
st.sidebar.info("""
### About
This app implements a Multi-Layer Perceptron (MLP) neural network for bird classification.
- Configure the network architecture and training parameters
- Train and test the model
- Analyze performance metrics
- Classify individual samples
""")

# Display current configuration summary
st.sidebar.markdown("---")
st.sidebar.subheader("Current Configuration")
st.sidebar.write(f"Hidden Layers: {num_hidden_layers}")
st.sidebar.write(f"Neurons: {neurons_per_layer}")
st.sidebar.write(f"Learning Rate: {learning_rate}")
st.sidebar.write(f"Epochs: {epochs}")
st.sidebar.write(f"Bias: {'Yes' if add_bias else 'No'}")
st.sidebar.write(f"Activation: {activation_function}")