import streamlit as st
import tensorflow as tf
import cv2
from datetime import datetime
from PIL import Image
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import pickle
import plotly.figure_factory as ff
from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array
import joblib


def main():
    # Initialize session state for user inputs
    if 'name' not in st.session_state:
        st.session_state['name'] = ''
    if 'age' not in st.session_state:
        st.session_state['age'] = 0
    if 'gender' not in st.session_state:
        st.session_state['gender'] = ''
    if 'birthdate' not in st.session_state:
        st.session_state['birthdate'] = datetime.today().date()
    if 'agreed' not in st.session_state:
        st.session_state['agreed'] = False


    # Set the logo in the sidebar
    st.sidebar.image("logo.png", use_column_width=True)

    # Add a title


    # Define the menu items
    # Define the menu items
    menu = ["Home", "Brain Tumor Classifier", "Diseases",
            'Diabetes Prediction', 'Heart disease Prediction',
            'Parkison Prediction',
            'Liver prediction',
            'Hepatitis prediction',
            'Lung Cancer Prediction',
            'Chronic Kidney prediction',
            ]

    # Check gender to conditionally add or remove "Breast Cancer Prediction"
    if st.session_state['gender'] in ['Female', 'Other']:
        menu.append('Breast Cancer Prediction')

    elif 'Breast Cancer Prediction' in menu:
        menu.remove('Breast Cancer Prediction')


    if not st.session_state['agreed']:

        st.title("Welcom to The login page ")

        # Add a text input for the user's name
        st.markdown("<p style='font-size: 22px;'>Please enter your name:</p>", unsafe_allow_html=True)
        st.session_state['name'] = st.text_input("", st.session_state['name'], max_chars=50, key='name_input')

        st.markdown("<p style='font-size: 22px;'>Please enter your age:</p>", unsafe_allow_html=True)
        st.session_state['age'] = st.number_input("", min_value=0, max_value=120, step=1,
                                                  value=st.session_state['age'])

        # Display gender selectbox with larger font size
        st.markdown("<p style='font-size: 22px;'>Please select your gender:</p>", unsafe_allow_html=True)
        st.session_state['gender'] = st.selectbox("", ["Male", "Female", "Other"],
                                                  index=["Male", "Female", "Other"].index(st.session_state['gender']) if
                                                  st.session_state['gender'] else 0)
        st.write("Birthdate:")
        st.write(st.session_state['birthdate'].strftime('%Y-%m-%d'))

        # Add an 'Agree' button
        if st.button("موافق"):
            if st.session_state['name'] and st.session_state['age'] and st.session_state['gender'] and st.session_state['birthdate']:
                st.session_state['agreed'] = True
                st.experimental_rerun()
            else:
                st.error("الرجاء إدخال جميع البيانات المطلوبة.")

    else:
        # Add a 'Back' button to return to the input page
        if st.button("رجوع"):
            st.session_state['agreed'] = False
            st.experimental_rerun()

        # Display the menu in the sidebar
        choice = st.sidebar.radio("Menu", menu)

        # Define the content for each menu item
        if choice == "Home":
            st.subheader("Home")
            st.write("Welcome to the Home page.")



        elif choice == "Brain Tumor Classifier":
            st.subheader("Brain Tumor Classifier")
            st.title('Brain Tumor Classifier')
            if st.button("Information and Input Names"):
                with st.expander("Input Feature Names and Descriptions"):
                    st.write("""
                                   Upload an image of a brain scan to classify whether it has a tumor, and if so, what type.
                                   """)
            # Display user details from session state
            with st.expander("User Details"):
                st.write("Name:", st.session_state.get('name', 'N/A'))
                st.write("Age:", st.session_state.get('age', 'N/A'))
                st.write("Gender:", st.session_state.get('gender', 'N/A'))

            st.write('')


            file = st.file_uploader(label='Upload image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False,
                                    key=None)
            IMAGE_SIZE = 150

            effnet = EfficientNetB0(weights=None, include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
            model1 = effnet.output
            model1 = tf.keras.layers.GlobalAveragePooling2D()(model1)
            model1 = tf.keras.layers.Dropout(0.5)(model1)
            model1 = tf.keras.layers.Dense(4, activation='softmax')(model1)
            model1 = tf.keras.models.Model(inputs=effnet.input, outputs=model1)
            model1.load_weights('effnet.h5')

            if file is not None:
                image = Image.open(file)
                image = np.array(image)
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                st.image(image, use_column_width=True)  # تغيير هنا
                images = image.reshape(1, 150, 150, 3)

            if st.button('تشخيص'):
                if file is not None:
                    predictions1 = model1.predict(images)
                    labels = ['No Tumor', 'Pituitary Tumor', 'Meningioma Tumor', 'Glioma Tumor']
                    st.write('Prediction over the uploaded image:')
                    disease_index = np.argmax(predictions1)
                    disease_name = labels[disease_index]

                    # Display user's name and predicted disease with specified font size
                    st.markdown(f"**Patient name:** {st.session_state['name']}")
                    st.markdown(f"**Predicted Disease:** {disease_name}")

                    # Plotting the predictions
                    fig, ax = plt.subplots()
                    y_pos = np.arange(len(labels))
                    ax.barh(y_pos, predictions1[0], align='center')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(labels)
                    ax.invert_yaxis()  # labels read top-to-bottom
                    ax.set_xlabel('Probability')
                    ax.set_title('Predictions')

                    # Show the plot in Streamlit
                    st.pyplot(fig)


        elif choice == "Diseases":
            st.subheader("Diseases")
            st.write("Welcome to the Diseases page!")
            if st.button("Information and Input Names"):
                with st.expander("Input Feature Names and Descriptions"):
                    st.write("""
                    **Symptom_1:** Main symptom experienced.
                    **Symptom_2:** Second symptom (if any).
                    **Symptom_3:** Third symptom (if any).
                    **Symptom_4:** Fourth symptom (if any).
                    """)

            # Display user details from session state
            with st.expander("User Details"):
                st.write("Name:", st.session_state.get('name', 'N/A'))
                st.write("Age:", st.session_state.get('age', 'N/A'))
                st.write("Gender:", st.session_state.get('gender', 'N/A'))
            model = CatBoostClassifier()
            model.load_model('model/model.cbm')

            data = pd.read_csv('datasets/dataset_diseases.csv', sep=';')
            desc = pd.read_csv('datasets/symptom_Description.csv', sep=',')
            prec = pd.read_csv('datasets/symptom_precaution.csv', sep=',')

            data = data.fillna('none')

            def predict_disease(input_data):
                prediction = model.predict([input_data])
                return prediction[0]

            st.title("Disease prediction model")
            st.image('Disease.png', use_column_width=True)
            st.write(
                "Enter symptoms and click 'Predict' for advice. Not a medical recommendation, consult a specialist")

            symptom_1 = st.selectbox('Select the main symptom', data['Symptom_1'].unique())
            symptom_2 = st.selectbox('Select a second symptom (if any)', data['Symptom_2'].unique())
            symptom_3 = st.selectbox('Select a third symptom (if any)', data['Symptom_3'].unique())
            symptom_4 = st.selectbox('Select a fourth symptom (if any)', data['Symptom_4'].unique())

            input_data = pd.Series([symptom_1, symptom_2, symptom_3, symptom_4])

            if st.button('Predict'):
                prediction = predict_disease(input_data)

                selected_desc = desc[desc['Disease'] == prediction[0]]['Description'].values[0]
                selected_prec = prec[prec['Disease'] == prediction[0]].iloc[:, 1:].values[0]

                st.write(f'<h3>Predicted disease: {prediction[0]}</h3>', unsafe_allow_html=True)
                st.write(f'<h3>Description: {selected_desc}</h3>', unsafe_allow_html=True)
                st.write('<h3>Recommendations:</h3>', unsafe_allow_html=True)
                for i, precaution in enumerate(selected_prec):
                    st.write(f'<h3>Precaution_{i + 1}: {precaution}</h3>', unsafe_allow_html=True)


        elif choice == 'Diabetes Prediction':
            # Title and image display
            st.title("Diabetes Disease Prediction")
            image = Image.open('d3.png')
            st.image(image, caption='Diabetes Disease Prediction')

            # Load the diabetes prediction model
            try:
                diabetes_model = joblib.load("models/diabetes_model.sav")
            except FileNotFoundError:
                st.error(
                    "Model file not found. Please ensure the model file 'diabetes_model.sav' is present in the 'models' directory.")
                st.stop()

            # Information and input feature names
            if st.button("Information and Input Names"):
                with st.expander("Input Feature Names and Descriptions"):
                    st.write("""
                        **Pregnancies:** Number of times pregnant (only applicable for females).
                        **Glucose:** Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
                        **BloodPressure:** Diastolic blood pressure (mm Hg).
                        **SkinThickness:** Triceps skinfold thickness (mm).
                        **Insulin:** 2-Hour serum insulin (mu U/ml).
                        **BMI:** Body mass index (weight in kg/(height in m)^2).
                        **DiabetesPedigreeFunction:** Diabetes pedigree function.
                    """)

            # Display user details from session state
            with st.expander("User Details"):
                name = st.session_state.get('name', 'N/A')
                age = st.session_state.get('age', 'N/A')
                gender = st.session_state.get('gender', 'N/A')
                st.write("Name:", name)
                st.write("Age:", age)
                st.write("Gender:", gender)

            # Gender-specific input
            gender = st.session_state.get('gender')

            if gender == 'Female':
                Pregnancies = st.selectbox("Number of Pregnancies", options=list(range(0, 11)))
            else:
                Pregnancies = 0

            # Collecting user input for other features
            Glucose = st.slider("Glucose Level", min_value=0, max_value=200, value=80)
            BloodPressure = st.slider("Blood Pressure Value", min_value=0, max_value=150, value=60)
            SkinThickness = st.slider("Skin Thickness Value", min_value=0, max_value=50, value=20)
            Insulin = st.slider("Insulin Value", min_value=0, max_value=500, value=80)
            BMI = st.slider("BMI Value", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
            DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function Value", min_value=0.0, max_value=10.0,
                                                 value=0.5, step=0.1)

            # Since age is not being collected anymore, we set it to 0
            Age = 0

            # Diabetes test result button
            if st.button("Diabetes Test Result"):
                try:
                    diabetes_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                                                   Insulin, BMI, DiabetesPedigreeFunction, Age]])

                    if diabetes_prediction[0] == 1:
                        result = 'Diabetic'
                        diabetes_dig = "We are really sorry to say but it seems like you are Diabetic."
                        try:
                            image = Image.open('positive.jpg')
                            st.image(image, caption='Positive Result')
                        except FileNotFoundError:
                            st.warning(
                                "Positive result image not found. Please ensure 'positive.jpg' is present in the directory.")
                    else:
                        result = 'Not Diabetic'
                        diabetes_dig = 'Congratulations, You are not diabetic.'
                        try:
                            image = Image.open('negative.jpg')
                            st.image(image, caption='Negative Result')
                        except FileNotFoundError:
                            st.warning(
                                "Negative result image not found. Please ensure 'negative.jpg' is present in the directory.")

                    st.success(f"{name}, {diabetes_dig}")

                    # Plotting the result with seaborn
                    labels = ['Diabetic', 'Not Diabetic']
                    sizes = [1 if result == 'Diabetic' else 0, 1 if result == 'Not Diabetic' else 0]
                    colors = sns.color_palette("coolwarm", 2)

                    fig, ax = plt.subplots()
                    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                                      startangle=90, textprops=dict(color="w"))

                    ax.legend(wedges, labels, title="Status", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                    plt.setp(autotexts, size=10, weight="bold")
                    ax.set_title("Diabetes Prediction Result")

                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error in prediction: {e}")
        elif choice == 'Heart disease Prediction':
            st.title("Heart Disease Prediction")

            # Load the heart disease prediction model
            heart_model = joblib.load("models/heart_disease_model.sav")

            if st.button("Information and Input Names"):
                with st.expander("Input Feature Names and Descriptions"):
                    st.write("""
                        
                        """)

            # Display user details from session state
            with st.expander("User Details"):
                name = st.session_state.get('name', 'N/A')
                age = st.session_state.get('age', 'N/A')
                gender = st.session_state.get('gender', 'N/A')
                st.write("Name:", name)
                st.write("Age:", age)
                st.write("Gender:", gender)

            # Organize input fields using sliders
            col1, col2, col3 = st.columns(3)

            # Column 1: Numerical inputs using sliders
            with col1:
                st.subheader("Numerical Inputs")
                trestbps = st.slider("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
                chol = st.slider("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
                thalach = st.slider("Max Heart Rate Achieved (bpm)", min_value=0, max_value=300, value=150)
                ca = st.slider("Number of Major Vessels (0–3) Colored by Flourosopy", min_value=0, max_value=3, value=0)

            # Column 2: Categorical inputs
            with col2:
                st.subheader("Categorical Inputs")
                cp = st.selectbox("Chest Pain Type",
                                  ["typical angina", "atypical angina", "non — anginal pain", "asymptotic"])
                restecg = st.selectbox("Resting ECG",
                                       ["normal", "having ST-T wave abnormality", "left ventricular hypertrophy"])
                slope = st.selectbox("Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])

            # Column 3: Additional inputs
            with col3:
                st.subheader("Additional Inputs")
                thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"])
                exang = st.checkbox('Exercise Induced Angina')
                fbs = st.checkbox('Fasting Blood Sugar > 120 mg/dl')

            # Form submission
            if st.button("Heart test result"):
                # Convert categorical features to numeric using one-hot encoding
                cp_encoded = 0 if cp == "typical angina" else (
                    1 if cp == "atypical angina" else (2 if cp == "non — anginal pain" else 3))
                restecg_encoded = 0 if restecg == "normal" else (1 if restecg == "having ST-T wave abnormality" else 2)
                slope_encoded = 0 if slope == "upsloping" else (1 if slope == "flat" else 2)
                thal_encoded = 0 if thal == "normal" else (1 if thal == "fixed defect" else 2)

                # Perform prediction
                features = [[age, 1 if gender == "Male" else 0, cp_encoded, trestbps, chol, 1 if fbs else 0,
                             restecg_encoded, thalach, 1 if exang else 0, 0, slope_encoded, ca, thal_encoded]]

                heart_prediction = heart_model.predict(features)

                # Display results
                if heart_prediction[0] == 1:
                    st.image('positive.jpg', caption='Positive Result')
                    st.error(f"{name}, We are really sorry to say but it seems like you have Heart Disease.")
                else:
                    st.image('negative.jpg', caption='Negative Result')
                    st.success(f"{name}, Congratulations, You don't have Heart Disease.")

                # Visualization
                labels = ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 'Fasting Blood Sugar',
                          'Resting ECG', 'Max Heart Rate', 'Exercise Induced Angina', 'ST Depression', 'Slope',
                          'Number of Vessels', 'Thalassemia']
                values = [age, 1 if gender == "Male" else 0, cp_encoded, trestbps, chol, 1 if fbs else 0,
                          restecg_encoded, thalach, 1 if exang else 0, 0, slope_encoded, ca, thal_encoded]

                fig = go.Figure(data=[go.Bar(x=labels, y=values)])
                fig.update_layout(title='Heart Disease Prediction Inputs',
                                  xaxis_title='Features',
                                  yaxis_title='Values')

                st.plotly_chart(fig)

        elif choice == 'Parkison Prediction':
            st.title("Parkinson Prediction")
            image = Image.open('p1.jpg')
            st.image(image, caption='Parkinson\'s disease')

            # Load the model
            parkinson_model = joblib.load("models/parkinsons_model.sav")

            # Button for displaying input information
            if st.button("Information and Input Names"):
                with st.expander("Input Feature Names and Descriptions"):
                    st.write("""
                    ### Input Feature Names and Descriptions:
                    - **MDVP:Fo(Hz)**: Average vocal fundamental frequency.
                    - **MDVP:Fhi(Hz)**: Maximum vocal fundamental frequency.
                    - **MDVP:Flo(Hz)**: Minimum vocal fundamental frequency.
                    - **MDVP:Jitter(%)**: Variation in fundamental frequency.
                    - **MDVP:Jitter(Abs)**: Variation in fundamental frequency (absolute).
                    - **MDVP:RAP**: Relative amplitude perturbation.
                    - **MDVP:PPQ**: Five-point period perturbation quotient.
                    - **Jitter:DDP**: Average absolute difference of differences between cycles.
                    - **MDVP:Shimmer**: Variation in amplitude.
                    - **MDVP:Shimmer(dB)**: Variation in amplitude in decibels.
                    - **Shimmer:APQ3**: Three-point amplitude perturbation quotient.
                    - **Shimmer:APQ5**: Five-point amplitude perturbation quotient.
                    - **MDVP:APQ**: Eleven-point amplitude perturbation quotient.
                    - **Shimmer:DDA**: Average absolute differences between cycles.
                    - **NHR**: Noise-to-harmonics ratio.
                    - **HNR**: Harmonics-to-noise ratio.
                    - **RPDE**: Recurrence period density entropy.
                    - **DFA**: Detrended fluctuation analysis.
                    - **spread1**: Nonlinear measure of fundamental frequency variation.
                    - **spread2**: Nonlinear measure of fundamental frequency variation.
                    - **D2**: Nonlinear dynamical complexity measure.
                    - **PPE**: Pitch period entropy.
                    """)

            # Display user details from session state
            with st.expander("User Details"):
                st.write("Name:", st.session_state.get('name', 'N/A'))
                st.write("Age:", st.session_state.get('age', 'N/A'))
                st.write("Gender:", st.session_state.get('gender', 'N/A'))

            # Collect input parameters
            st.header("Enter the following details:")

            col1, col2, col3 = st.columns(3)

            with col1:
                MDVP = st.slider("MDVP:Fo(Hz)", 0.0, 300.0, step=0.1)
                MDVPJITTER = st.slider("MDVP:Jitter(%)", 0.0, 1.0, step=0.001)
                MDVPShimmer = st.slider("MDVP:Shimmer", 0.0, 1.0, step=0.001)
                ShimmerAPQ5 = st.slider("Shimmer:APQ5", 0.0, 1.0, step=0.001)
                spread1 = st.slider("spread1", -10.0, 10.0, step=0.1)
                spread2 = st.slider("spread2", -10.0, 10.0, step=0.1)
                PPE = st.slider("PPE", 0.0, 1.0, step=0.001)

            with col2:
                MDVPFIZ = st.slider("MDVP:Fhi(Hz)", 0.0, 600.0, step=0.1)
                MDVPJitterAbs = st.slider("MDVP:Jitter(Abs)", 0.0, 0.1, step=0.00001)
                MDVPShimmer_dB = st.slider("MDVP:Shimmer(dB)", 0.0, 10.0, step=0.01)
                MDVP_APQ = st.slider("MDVP:APQ", 0.0, 1.0, step=0.001)
                NHR = st.slider("NHR", 0.0, 1.0, step=0.001)
                HNR = st.slider("HNR", 0.0, 50.0, step=0.1)
                RPDE = st.slider("RPDE", 0.0, 1.0, step=0.001)

            with col3:
                MDVPFLO = st.slider("MDVP:Flo(Hz)", 0.0, 300.0, step=0.1)
                MDVPRAP = st.slider("MDVP:RAP", 0.0, 1.0, step=0.001)
                Shimmer_APQ3 = st.slider("Shimmer:APQ3", 0.0, 1.0, step=0.001)
                ShimmerDDA = st.slider("Shimmer:DDA", 0.0, 1.0, step=0.001)
                DFA = st.slider("DFA", 0.0, 2.0, step=0.01)
                D2 = st.slider("D2", 0.0, 5.0, step=0.01)
                MDVPPPQ = st.slider("MDVP:PPQ", 0.0, 1.0, step=0.001)
                JitterDDP = st.slider("Jitter:DDP", 0.0, 0.1, step=0.001)

            # Code for prediction
            parkinson_dig = ''

            # Button for prediction
            if st.button("Parkinson Test Result"):
                # Prepare input features for the model
                input_features = [
                    MDVP, MDVPFIZ, MDVPFLO, MDVPJITTER, MDVPJitterAbs, MDVPRAP,
                    MDVPPPQ, JitterDDP, MDVPShimmer, MDVPShimmer_dB,
                    Shimmer_APQ3, ShimmerAPQ5, MDVP_APQ, ShimmerDDA, NHR, HNR,
                    RPDE, DFA, spread1, spread2, D2, PPE
                ]

                # Make prediction
                parkinson_prediction = parkinson_model.predict([input_features])

                # Display result
                if parkinson_prediction[0] == 1:
                    parkinson_dig = 'We are really sorry to say but it seems like you have Parkinson\'s disease.'
                    result_image = Image.open('positive.jpg')
                else:
                    parkinson_dig = "Congratulations, You don't have Parkinson's disease."
                    result_image = Image.open('negative.jpg')

                st.image(result_image, caption='')
                st.success(f'{st.session_state["name"]}, {parkinson_dig}')

                # Display plot
                fig, ax = plt.subplots()
                ax.bar(['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
                        'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
                        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'], input_features)
                ax.set_xlabel('Features')
                ax.set_ylabel('Values')
                ax.set_title('Parkinson Test Input Features')
                plt.xticks(rotation=90)
                st.pyplot(fig)


        elif choice == 'Lung Cancer Prediction':
            # Load the lung cancer prediction model
            lung_cancer_model = joblib.load('models/lung_cancer_model.sav')

            # Title of the app
            st.title("Lung Cancer Prediction")

            # Load and preprocess data
            lung_cancer_data = pd.read_csv('data/lung_cancer.csv')
            lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].map({'M': 'Male', 'F': 'Female'})

            # Display the image
            image = Image.open('h.JPG')
            st.image(image, caption='Lung Cancer Prediction')

            # Information and Input Names button
            if st.button("Information and Input Names"):
                with st.expander("Input Feature Names and Descriptions"):
                    st.write("""
                    - GENDER: The gender of the patient (Male/Female)
                    - AGE: The age of the patient
                    - SMOKING: Whether the patient smokes or not (YES/NO)
                    - YELLOW_FINGERS: Whether the patient has yellow fingers (YES/NO)
                    - ANXIETY: Whether the patient has anxiety (YES/NO)
                    - PEER_PRESSURE: Whether the patient is under peer pressure (YES/NO)
                    - CHRONICDISEASE: Whether the patient has a chronic disease (YES/NO)
                    - FATIGUE: Whether the patient feels fatigue (YES/NO)
                    - ALLERGY: Whether the patient has any allergies (YES/NO)
                    - WHEEZING: Whether the patient experiences wheezing (YES/NO)
                    - ALCOHOLCONSUMING: Whether the patient consumes alcohol (YES/NO)
                    - COUGHING: Whether the patient has a cough (YES/NO)
                    - SHORTNESSOFBREATH: Whether the patient experiences shortness of breath (YES/NO)
                    - SWALLOWINGDIFFICULTY: Whether the patient has difficulty swallowing (YES/NO)
                    - CHESTPAIN: Whether the patient experiences chest pain (YES/NO)
                    """)

            # Display user details from session state
            with st.expander("User Details"):
                st.write(f"Name: {st.session_state.get('name', 'N/A')}")
                st.write(f"Age: {st.session_state.get('age', 'N/A')}")
                st.write(f"Gender: {st.session_state.get('gender', 'N/A')}")

            # Improved input layout
            st.subheader("Please provide the following information:")

            col1, col2, col3 = st.columns(3)

            with col1:
                smoking = st.selectbox("Smoking:", ['NO', 'YES'])
                peer_pressure = st.selectbox("Peer Pressure:", ['NO', 'YES'])
                allergy = st.selectbox("Allergy:", ['NO', 'YES'])
                coughing = st.selectbox("Coughing:", ['NO', 'YES'])
                chest_pain = st.selectbox("Chest Pain:", ['NO', 'YES'])

            with col2:
                yellow_fingers = st.selectbox("Yellow Fingers:", ['NO', 'YES'])
                chronic_disease = st.selectbox("Chronic Disease:", ['NO', 'YES'])
                wheezing = st.selectbox("Wheezing:", ['NO', 'YES'])
                shortness_of_breath = st.selectbox("Shortness of Breath:", ['NO', 'YES'])

            with col3:
                anxiety = st.selectbox("Anxiety:", ['NO', 'YES'])
                fatigue = st.selectbox("Fatigue:", ['NO', 'YES'])
                alcohol_consuming = st.selectbox("Alcohol Consuming:", ['NO', 'YES'])
                swallowing_difficulty = st.selectbox("Swallowing Difficulty:", ['NO', 'YES'])

            # Prediction button and result
            cancer_result = ''
            if st.button("Predict Lung Cancer"):
                user_data = pd.DataFrame({
                    'GENDER': [st.session_state.get('gender')],
                    'AGE': [st.session_state.get('age')],
                    'SMOKING': [smoking],
                    'YELLOW_FINGERS': [yellow_fingers],
                    'ANXIETY': [anxiety],
                    'PEER_PRESSURE': [peer_pressure],
                    'CHRONICDISEASE': [chronic_disease],
                    'FATIGUE': [fatigue],
                    'ALLERGY': [allergy],
                    'WHEEZING': [wheezing],
                    'ALCOHOLCONSUMING': [alcohol_consuming],
                    'COUGHING': [coughing],
                    'SHORTNESSOFBREATH': [shortness_of_breath],
                    'SWALLOWINGDIFFICULTY': [swallowing_difficulty],
                    'CHESTPAIN': [chest_pain]
                })

                # Map string values to numeric
                user_data.replace({'NO': 1, 'YES': 2}, inplace=True)

                # Convert columns to numeric where necessary
                numeric_columns = ['AGE', 'FATIGUE', 'ALLERGY', 'ALCOHOLCONSUMING', 'COUGHING', 'SHORTNESSOFBREATH']
                user_data[numeric_columns] = user_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

                # Perform prediction
                cancer_prediction = lung_cancer_model.predict(user_data)

                # Determine the risk score based on prediction (hypothetical example)
                risk_score = lung_cancer_model.predict_proba(user_data)[:, 1]  # Adjust according to model output

                # Display result
                if cancer_prediction[0] == 'YES':
                    cancer_result = "there is a risk of Lung Cancer."
                    st.image('positive.jpg', caption='')
                else:
                    cancer_result = " no significant risk of Lung Cancer."
                    st.image('negative.jpg', caption='')

                st.success(f"{st.session_state.get('name', 'User')}, {cancer_result}")

                # Display risk score chart
                st.subheader("Risk Score Visualization")
                st.bar_chart({"Risk Score": [risk_score]})

            # Liver prediction page
        elif choice == 'Liver prediction':
            st.title("Liver Disease Prediction")

            # Display image for liver disease prediction
            image = Image.open('liver.jpg')
            st.image(image, caption='Liver disease prediction.')

            # Load the liver model
            liver_model = joblib.load('models/liver_model.sav')

            if st.button("Information and Input Names"):
                with st.expander("Input Feature Names and Descriptions"):
                    st.write("""
                               - Total_Bilirubin: This is a measure of the amount of bilirubin in the blood.
                               - Direct_Bilirubin: This is a measure of the direct (conjugated) bilirubin in the blood.
                               - Alkaline_Phosphotase: This is an enzyme found in the liver and bones.
                               - Alamine_Aminotransferase: Also known as ALT, this enzyme is found in the liver.
                               - Aspartate_Aminotransferase: Also known as AST, this enzyme is found in various tissues including the liver and muscles.
                               - Total_Proteins: Total amount of proteins in the blood.
                               - Albumin: This is a protein made by the liver.
                               - Albumin_and_Globulin_Ratio: Ratio of albumin to globulin in the blood.
                           """)

            # Display user details from session state
            with st.expander("User Details"):
                name = st.session_state.get('name', 'N/A')
                age = st.session_state.get('age', 'N/A')
                gender_value = st.session_state.get('gender', 'N/A')
                st.write("Name:", name)
                st.write("Age:", age)
                st.write("Gender:", gender_value)

                if gender_value is None:
                    st.error("Please enter your gender on the main page.")

            # Create input columns
            col1, col2, col3 = st.columns(3)

            with col1:
                Sex = 0 if gender_value == "Male" else 1

            Total_Bilirubin = col3.slider("Total Bilirubin", min_value=0.0, max_value=50.0, step=0.1)
            Direct_Bilirubin = col1.slider("Direct Bilirubin", min_value=0.0, max_value=10.0, step=0.1)
            Alkaline_Phosphotase = col2.slider("Alkaline Phosphotase", min_value=0, max_value=1000, step=1)
            Alamine_Aminotransferase = col3.slider("Alamine Aminotransferase", min_value=0, max_value=500, step=1)
            Aspartate_Aminotransferase = col1.slider("Aspartate Aminotransferase", min_value=0, max_value=500, step=1)
            Total_Proteins = col2.slider("Total Proteins", min_value=0.0, max_value=10.0, step=0.1)
            Albumin = col3.slider("Albumin", min_value=0.0, max_value=6.0, step=0.1)
            Albumin_and_Globulin_Ratio = col1.slider("Albumin and Globulin Ratio", min_value=0.0, max_value=3.0,
                                                     step=0.1)

            # Button for liver test result
            if st.button("Liver Test Result"):
                if gender_value is None:
                    st.error("Please enter your gender on the main page.")
                else:
                    liver_prediction = liver_model.predict(
                        [[Sex, age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
                          Alamine_Aminotransferase, Aspartate_Aminotransferase,
                          Total_Proteins, Albumin, Albumin_and_Globulin_Ratio]])

                    if liver_prediction[0] == 1:
                        image = Image.open('positive.jpg')
                        st.image(image, caption='')
                        liver_dig = "We are really sorry to say, but it seems like you have liver disease."
                    else:
                        image = Image.open('negative.jpg')
                        st.image(image, caption='')
                        liver_dig = "Congratulations, you don't have liver disease."

                    st.success(f'{name}, {liver_dig}')

                    # Plotting the input values
                    fig, ax = plt.subplots()
                    labels = ['Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                              'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',
                              'Total_Proteins', 'Albumin', 'Albumin_and_Globulin_Ratio']
                    values = [Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
                              Alamine_Aminotransferase, Aspartate_Aminotransferase,
                              Total_Proteins, Albumin, Albumin_and_Globulin_Ratio]

                    ax.barh(labels, values, color='skyblue')
                    ax.set_xlabel('Values')
                    ax.set_title('Liver Function Test Results')

                    st.pyplot(fig)

        elif choice == 'Hepatitis prediction':
            st.title("Hepatitis Prediction")
            hepatitis_model = joblib.load('models/hepititisc_model.sav')
            # Display an image related to Hepatitis prediction
            image = Image.open('m.jpg')
            st.image(image, caption='Hepatitis Prediction')

            # Button to show information and input names
            if st.button("Information and Input Names"):
                with st.expander("Input Feature Names and Descriptions"):
                    st.write("""
                           - Age: Age of the patient
                           - Sex: Gender of the patient (Male: 1, Female: 2)
                           - Total Bilirubin: Level of total bilirubin in the blood
                           - Direct Bilirubin: Level of direct bilirubin in the blood
                           - Alkaline Phosphatase: Level of alkaline phosphatase enzyme in the blood
                           - Alamine Aminotransferase: Level of alanine aminotransferase enzyme in the blood
                           - Aspartate Aminotransferase: Level of aspartate aminotransferase enzyme in the blood
                           - Total Proteins: Total protein level in the blood
                           - Albumin: Albumin level in the blood
                           - Albumin and Globulin Ratio: Ratio of albumin to globulin in the blood
                           - GGT: Gamma-glutamyl transferase level in the blood
                           - PROT: Total protein level in the blood
                       """)

            # Display user details from session state
            with st.expander("User Details"):
                name = st.session_state.get('name', 'N/A')
                age = st.session_state.get('age', 'N/A')
                gender = st.session_state.get('gender', 'N/A')
                st.write("Name:", name)
                st.write("Age:", age)
                st.write("Gender:", gender)
            sex = 1 if gender == "Male" else 2

            # Columns for user input
            col1, col2, col3 = st.columns(3)

            with col1:
                total_bilirubin = st.slider("Total Bilirubin", 0.0, 50.0, 0.1)
                direct_bilirubin = st.slider("Direct Bilirubin", 0.0, 50.0, 0.1)
                aspartate_aminotransferase = st.slider("Aspartate Aminotransferase", 0.0, 2000.0, 1.0)
                albumin_and_globulin_ratio = st.slider("Albumin and Globulin Ratio", 0.0, 3.0, 0.1)

            with col2:
                alkaline_phosphatase = st.slider("Alkaline Phosphatase", 0.0, 2000.0, 1.0)
                total_proteins = st.slider("Total Proteins", 0.0, 10.0, 0.1)
                your_ggt_value = st.slider("GGT", 0.0, 500.0, 1.0)

            with col3:
                alamine_aminotransferase = st.slider("Alamine Aminotransferase", 0.0, 2000.0, 1.0)
                albumin = st.slider("Albumin", 0.0, 10.0, 0.1)
                your_prot_value = st.slider("PROT", 0.0, 100.0, 1.0)

            # Button for prediction
            if st.button("Predict Hepatitis"):
                # Create a DataFrame with user inputs
                user_data = pd.DataFrame({
                    'Age': [age],
                    'Sex': [sex],
                    'Total Bilirubin': [total_bilirubin],
                    'Direct Bilirubin': [direct_bilirubin],
                    'Alkaline Phosphatase': [alkaline_phosphatase],
                    'Alamine Aminotransferase': [alamine_aminotransferase],
                    'Aspartate Aminotransferase': [aspartate_aminotransferase],
                    'Total Proteins': [total_proteins],
                    'Albumin': [albumin],
                    'Albumin and Globulin Ratio': [albumin_and_globulin_ratio],
                    'GGT': [your_ggt_value],
                    'PROT': [your_prot_value]
                })

                # Perform prediction
                hepatitis_prediction = hepatitis_model.predict(user_data)

                # Display result
                if hepatitis_prediction[0] == 1:
                    st.error(f"{name}, We are really sorry to say but it seems like you have Hepatitis.")
                    image = Image.open('positive.jpg')
                    st.image(image, caption='Hepatitis Positive')
                else:
                    st.success(f"{name}, Congratulations, you do not have Hepatitis.")
                    image = Image.open('negative.jpg')
                    st.image(image, caption='Hepatitis Negative')

                # Plotting user input data
                st.write("### User Input Data Visualization")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=user_data.columns, y=user_data.iloc[0], palette="viridis", ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_title("User Input Features")
                ax.set_ylabel("Values")
                ax.set_xlabel("Features")
                st.pyplot(fig)

        elif choice == 'Chronic Kidney prediction':
            st.title("Chronic Kidney Disease Prediction")

            # Load the model
            chronic_disease_model = joblib.load('models/chronic_model.sav')

            if st.button("Information and Input Names"):
                with st.expander("Input Feature Names and Descriptions"):
                    st.write("""
                            Add relevant information and descriptions of input features here.
                            Example:
                            - **Blood Pressure**: Systolic blood pressure measured in mmHg.
                            - **Albumin**: Presence of albumin in urine, measured in g/dL, etc.
                        """)

                # Display user details from session state
            with st.expander("User Details"):
                name = st.session_state.get('name', 'N/A')
                age = st.session_state.get('age', 'N/A')
                gender = st.session_state.get('gender', 'N/A')
                st.write("Name:", name)
                st.write("Age:", age)
                st.write("Gender:", gender)

                # User input for features
            col1, col2, col3 = st.columns(3)
            with col1:
                bp = st.slider("Blood Pressure", 50, 200, 120)
                al = st.slider("Albumin", 0, 5, 0)
                pc = st.selectbox("Pus Cells", ["Normal", "Abnormal"])
                pc = 1 if pc == "Normal" else 0
                bgr = st.slider("Blood Glucose Random", 50, 200, 120)
                sod = st.slider("Sodium", 100, 200, 140)
                pcv = st.slider("Packed Cell Volume", 20, 60, 40)
                htn = st.selectbox("Hypertension", ["Yes", "No"])
                htn = 1 if htn == "Yes" else 0
                appet = st.selectbox("Appetite", ["Good", "Poor"])
                appet = 1 if appet == "Good" else 0

            with col2:
                sg = st.slider("Specific Gravity", 1.0, 1.05, 1.02)
                su = st.slider("Sugar", 0, 5, 0)
                pcc = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"])
                pcc = 1 if pcc == "Present" else 0
                bu = st.slider("Blood Urea", 10, 200, 60)
                pot = st.slider("Potassium", 2, 7, 4)
                wc = st.slider("White Blood Cell Count", 2000, 20000, 10000)
                dm = st.selectbox("Diabetes Mellitus", ["Yes", "No"])
                dm = 1 if dm == "Yes" else 0
                pe = st.selectbox("Pedal Edema", ["Yes", "No"])
                pe = 1 if pe == "Yes" else 0

            with col3:
                rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
                rbc = 1 if rbc == "Normal" else 0
                ba = st.selectbox("Bacteria", ["Present", "Not Present"])
                ba = 1 if ba == "Present" else 0
                sc = st.slider("Serum Creatinine", 0, 10, 3)
                hemo = st.slider("Hemoglobin", 3, 17, 12)
                rc = st.slider("Red Blood Cell Count", 2, 8, 4)
                cad = st.selectbox("Coronary Artery Disease", ["Yes", "No"])
                cad = 1 if cad == "Yes" else 0
                ane = st.selectbox("Anemia", ["Yes", "No"])
                ane = 1 if ane == "Yes" else 0

            # Button for prediction
            if st.button("Predict Chronic Kidney Disease"):
                # Create DataFrame with user inputs
                user_input = pd.DataFrame({
                    'age': [age],
                    'bp': [bp],
                    'sg': [sg],
                    'al': [al],
                    'su': [su],
                    'rbc': [rbc],
                    'pc': [pc],
                    'pcc': [pcc],
                    'ba': [ba],
                    'bgr': [bgr],
                    'bu': [bu],
                    'sc': [sc],
                    'sod': [sod],
                    'pot': [pot],
                    'hemo': [hemo],
                    'pcv': [pcv],
                    'wc': [wc],
                    'rc': [rc],
                    'htn': [htn],
                    'dm': [dm],
                    'cad': [cad],
                    'appet': [appet],
                    'pe': [pe],
                    'ane': [ane]
                })

                # Perform prediction
                kidney_prediction = chronic_disease_model.predict(user_input)

                # Display result
                if kidney_prediction[0] == 1:
                    st.error(f"{name}, we are really sorry to say but it seems like you have kidney disease.")

                    # Example plot for visualization
                    plt.figure(figsize=(8, 6))
                    labels = ['Blood Pressure', 'Albumin', 'Blood Glucose', 'Serum Creatinine']
                    values = [bp, al, bgr, sc]
                    plt.bar(labels, values, color='red')
                    plt.title('Factors Leading to Kidney Disease')
                    plt.xlabel('Factors')
                    plt.ylabel('Values')
                    st.pyplot(plt)

                    # Example image display
                    image = Image.open('positive.jpg')
                    st.image(image, caption='Positive Diagnosis')

                else:
                    st.success(f"{name}, congratulations! You don't have kidney disease.")

                    # Example plot for visualization
                    plt.figure(figsize=(8, 6))
                    labels = ['Blood Pressure', 'Albumin', 'Blood Glucose', 'Serum Creatinine']
                    values = [bp, al, bgr, sc]
                    plt.bar(labels, values, color='green')
                    plt.title('Factors for Healthy Kidneys')
                    plt.xlabel('Factors')
                    plt.ylabel('Values')
                    st.pyplot(plt)

                    # Example image display
                    image = Image.open('negative.jpg')
                    st.image(image, caption='Negative Diagnosis')

        elif choice == 'Breast Cancer Prediction':
            st.title("Breast Cancer Prediction")

            breast_cancer_model = joblib.load('models/breast_cancer.sav')

            if st.button("Information and Input Names"):
                with st.expander("Input Feature Names and Descriptions"):
                    st.write("""
                            - **radius_mean**: Mean of distances from the center to points on the perimeter.
                            - **texture_mean**: Standard deviation of gray-scale values.
                            - **perimeter_mean**: Mean size of the core tumor.
                            - **area_mean**: Mean area of the tumor.
                            - **smoothness_mean**: Mean of local variation in radius lengths.
                            - **compactness_mean**: Mean of perimeter^2 / area - 1.0.
                            - **concavity_mean**: Mean of severity of concave portions of the contour.
                            - **concave points_mean**: Mean of number of concave portions of the contour.
                            - **symmetry_mean**: Mean of symmetry of the tumor.
                            - **fractal_dimension_mean**: Mean of “coastline approximation” - 1.
                            - **radius_se**: Standard error for the mean of distances from the center to points on the perimeter.
                            - **texture_se**: Standard error of gray-scale values.
                            - **perimeter_se**: Standard error of the perimeter size.
                            - **area_se**: Standard error of the tumor area.
                            - **smoothness_se**: Standard error of local variation in radius lengths.
                            - **compactness_se**: Standard error of perimeter^2 / area - 1.0.
                            - **concavity_se**: Standard error of the severity of concave portions of the contour.
                            - **concave points_se**: Standard error of the number of concave portions of the contour.
                            - **symmetry_se**: Standard error of the symmetry of the tumor.
                            - **fractal_dimension_se**: Standard error of “coastline approximation” - 1.
                            - **radius_worst**: Worst (mean of the three largest values) for the mean of distances from the center to points on the perimeter.
                            - **texture_worst**: Worst (mean of the three largest values) for gray-scale values.
                            - **perimeter_worst**: Worst perimeter size.
                            - **area_worst**: Worst area size.
                            - **smoothness_worst**: Worst local variation in radius lengths.
                            - **compactness_worst**: Worst perimeter^2 / area - 1.0.
                            - **concavity_worst**: Worst severity of concave portions of the contour.
                            - **concave points_worst**: Worst number of concave portions of the contour.
                            - **symmetry_worst**: Worst symmetry of the tumor.
                            - **fractal_dimension_worst**: Worst “coastline approximation” - 1.
                        """)

                # Display user details from session state
            with st.expander("User Details"):
                name = st.session_state.get('name', 'N/A')
                age = st.session_state.get('age', 'N/A')
                gender = st.session_state.get('gender', 'N/A')
                st.write("Name:", name)
                st.write("Age:", age)
                st.write("Gender:", gender)

                # Layout for input sliders
            col1, col2, col3 = st.columns(3)

            with col1:
                radius_mean = st.slider("Enter your Radius Mean", 6.0, 30.0, 15.0)
                texture_mean = st.slider("Enter your Texture Mean", 9.0, 40.0, 20.0)
                perimeter_mean = st.slider("Enter your Perimeter Mean", 43.0, 190.0, 90.0)

            with col2:
                area_mean = st.slider("Enter your Area Mean", 143.0, 2501.0, 750.0)
                smoothness_mean = st.slider("Enter your Smoothness Mean", 0.05, 0.25, 0.1)
                compactness_mean = st.slider("Enter your Compactness Mean", 0.02, 0.3, 0.15)

            with col3:
                concavity_mean = st.slider("Enter your Concavity Mean", 0.0, 0.5, 0.2)
                concave_points_mean = st.slider("Enter your Concave Points Mean", 0.0, 0.2, 0.1)
                symmetry_mean = st.slider("Enter your Symmetry Mean", 0.1, 1.0, 0.5)

            with col1:
                fractal_dimension_mean = st.slider("Enter your Fractal Dimension Mean", 0.01, 0.1, 0.05)
                radius_se = st.slider("Enter your Radius SE", 0.1, 3.0, 1.0)
                texture_se = st.slider("Enter your Texture SE", 0.2, 2.0, 1.0)

            with col2:
                perimeter_se = st.slider("Enter your Perimeter SE", 1.0, 30.0, 10.0)
                area_se = st.slider("Enter your Area SE", 6.0, 500.0, 150.0)
                smoothness_se = st.slider("Enter your Smoothness SE", 0.001, 0.03, 0.01)

            with col3:
                compactness_se = st.slider("Enter your Compactness SE", 0.002, 0.2, 0.1)
                concavity_se = st.slider("Enter your Concavity SE", 0.0, 0.05, 0.02)
                concave_points_se = st.slider("Enter your Concave Points SE", 0.0, 0.03, 0.01)

            with col1:
                symmetry_se = st.slider("Enter your Symmetry SE", 0.1, 1.0, 0.5)
                fractal_dimension_se = st.slider("Enter your Fractal Dimension SE", 0.01, 0.1, 0.05)

            with col2:
                radius_worst = st.slider("Enter your Radius Worst", 7.0, 40.0, 20.0)
                texture_worst = st.slider("Enter your Texture Worst", 12.0, 50.0, 25.0)
                perimeter_worst = st.slider("Enter your Perimeter Worst", 50.0, 250.0, 120.0)

            with col3:
                area_worst = st.slider("Enter your Area Worst", 185.0, 4250.0, 1500.0)
                smoothness_worst = st.slider("Enter your Smoothness Worst", 0.07, 0.3, 0.15)
                compactness_worst = st.slider("Enter your Compactness Worst", 0.03, 0.6, 0.3)

            with col1:
                concavity_worst = st.slider("Enter your Concavity Worst", 0.0, 0.8, 0.4)
                concave_points_worst = st.slider("Enter your Concave Points Worst", 0.0, 0.2, 0.1)
                symmetry_worst = st.slider("Enter your Symmetry Worst", 0.1, 1.0, 0.5)

            with col2:
                fractal_dimension_worst = st.slider("Enter your Fractal Dimension Worst", 0.01, 0.2, 0.1)

            # Prediction button and result display
            breast_cancer_result = ''

            if st.button("Predict Breast Cancer"):
                # Create a DataFrame with user inputs
                user_input = pd.DataFrame({
                    'radius_mean': [radius_mean],
                    'texture_mean': [texture_mean],
                    'perimeter_mean': [perimeter_mean],
                    'area_mean': [area_mean],
                    'smoothness_mean': [smoothness_mean],
                    'compactness_mean': [compactness_mean],
                    'concavity_mean': [concavity_mean],
                    'concave points_mean': [concave_points_mean],
                    'symmetry_mean': [symmetry_mean],
                    'fractal_dimension_mean': [fractal_dimension_mean],
                    'radius_se': [radius_se],
                    'texture_se': [texture_se],
                    'perimeter_se': [perimeter_se],
                    'area_se': [area_se],
                    'smoothness_se': [smoothness_se],
                    'compactness_se': [compactness_se],
                    'concavity_se': [concavity_se],
                    'concave points_se': [concave_points_se],
                    'symmetry_se': [symmetry_se],
                    'fractal_dimension_se': [fractal_dimension_se],
                    'radius_worst': [radius_worst],
                    'texture_worst': [texture_worst],
                    'perimeter_worst': [perimeter_worst],
                    'area_worst': [area_worst],
                    'smoothness_worst': [smoothness_worst],
                    'compactness_worst': [compactness_worst],
                    'concavity_worst': [concavity_worst],
                    'concave points_worst': [concave_points_worst],
                    'symmetry_worst': [symmetry_worst],
                    'fractal_dimension_worst': [fractal_dimension_worst],
                })

                # Perform prediction
                breast_cancer_prediction = breast_cancer_model.predict(user_input)

                # Display result
                if breast_cancer_prediction[0] == 1:
                    image = Image.open('positive.jpg')
                    st.image(image, caption='Positive Result')
                    breast_cancer_result = "The model predicts that you have Breast Cancer."
                else:
                    image = Image.open('negative.jpg')
                    st.image(image, caption='Negative Result')
                    breast_cancer_result = "The model predicts that you don't have Breast Cancer."

                st.success(breast_cancer_result)

                # Plot a heatmap for the input values as an example of an illustrative plot
                st.subheader("Feature Values Heatmap")
                fig, ax = plt.subplots()
                sns.heatmap(user_input, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)



# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f8f9fa;
    }

    .css-1lcbmhc.e1fqkh3o1 {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .css-1d391kg {
        color: #333;
    }

    .css-1lcbmhc.e1fqkh3o1 .css-vubbuv {
        margin-bottom: 20px;
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }

    .css-1lcbmhc.e1fqkh3o1 .css-vubbuv .css-nlntq9 {
        list-style-type: none;
        padding: 0;
        margin: 0;
    }

    .css-1lcbmhc.e1fqkh3o1 .css-vubbuv .css-nlntq9 a {
        display: block;
        padding: 10px;
        color: #333;
        text-decoration: none;
        transition: background-color 0.3s;
        border-radius: 8px;
        margin-bottom: 8px;
    }

    .css-1lcbmhc.e1fqkh3o1 .css-vubbuv .css-nlntq9 a:hover,
    .css-1lcbmhc.e1fqkh3o1 .css-vubbuv .css-nlntq9 a:active {
        background-color: #f0f0f0;
    }

    .stContainer {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()
