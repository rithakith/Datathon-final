# Personalized Travel Recommendations - README

## Introduction

This project is a **Personalized Place Recommendation System** built using **Streamlit** and machine learning algorithms. The system provides users with customized travel recommendations based on their preferences for activities and destinations in Sri Lanka. It uses user input data along with a dataset of places and activities to suggest top destinations. Additionally, it sends the recommendations to the user via email.

## Features
1. **User Input Form**: Allows users to input their name, email, preferred activities, and bucket list destinations.
2. **Activity Matching**: Recommends places based on activity similarity using **TF-IDF** vectorization and **cosine similarity**.
3. **Place Rating**: Prioritizes recommendations based on place ratings, reviews, and user preferences.
4. **Machine Learning**: Uses a **Random Forest Classifier** to predict the top 5 recommended places for the user.
5. **Email Notifications**: Sends personalized travel recommendations to the user's email.
6. **Dynamic User Data Management**: Updates the user data CSV file with new information from each user session.

## Project Structure

```bash
.
├── user_data_version_3_10K_Users.csv
├── updated_places_with_activities.csv
├── personalized_travel_recommendations.py
├── random_forest_model.pkl
├── README.md
└── requirements.txt
```

### Files
1. `personalized_travel_recommendations.py`: The main application script that runs the Streamlit app.
2. `user_data_version_3_10K_Users.csv`: Dataset containing user data, including user preferences and travel history.
3. `updated_places_with_activities.csv`: Dataset with place details and activities.
4. `random_forest_model.pkl`: The serialized Random Forest model for predicting recommended places.
5. `README.md`: This README file for the project.
6. `requirements.txt`: Contains a list of required Python packages for the project.

## Datasets
- **User Data** (`user_data_version_3_10K_Users.csv`): Contains user information such as name, email, preferred activities, and bucket list destinations.
- **Places Data** (`updated_places_with_activities.csv`): Contains place details, including ratings, available activities, and review data.

## How It Works

1. **Load Data**: The app loads user data and place data from CSV files.
2. **User Form**: Users enter their travel preferences, including activities they enjoy and bucket list destinations in Sri Lanka.
3. **Activity Processing**: The user’s preferred activities are matched against available activities at different places using **TF-IDF** vectorization and **cosine similarity**.
4. **Random Forest Classifier**: A **Random Forest Classifier** predicts the likelihood of a user recommending a place based on activity similarity, ratings, and review counts.
5. **Recommendations**: The top 5 recommended places are displayed and sent to the user via email.
6. **Data Update**: The user data is updated in the CSV file to record new preferences.

## Setup Instructions

### Prerequisites

- Python 3.x
- Streamlit
- Scikit-learn
- Pandas

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo-url.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd personalized_travel_recommendations
    ```

3. **Install required dependencies**:
    You can use `pip` to install the necessary packages. Run the following command:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit App**:
    Run the following command in the terminal to launch the app:
    ```bash
    streamlit run personalized_travel_recommendations.py
    ```

5. **Prepare Your Gmail Account for Email Sending**:
    - You need to set up an **App Password** in Gmail to send emails via the script. Follow [these instructions](https://support.google.com/accounts/answer/185833?hl=en) to create an app password and replace it in the code at `sender_password`.

## Usage Instructions

1. After launching the app, fill in your **name**, **email**, and select the **activities** you are interested in from the multiselect field.
2. Optionally, you can add some **bucket list destinations** in Sri Lanka to further personalize the recommendations.
3. Click the **"Get Recommendations"** button.
4. You will be shown a list of top 5 recommended places on the app interface.
5. The same recommendations will be emailed to you at the email address you provided.

## Technical Details

1. **TF-IDF & Cosine Similarity**: These are used to compute activity similarity between the user’s preferences and the available activities at different places.
2. **Random Forest Classifier**: Trained on place ratings, reviews, and activity similarity to predict the top places for the user.
3. **Data Processing**: User data and place data are cleaned and pre-processed for model input, including feature extraction and normalization.
4. **Email Sending**: The email functionality is built using Python’s `smtplib` and `email` libraries.

## Email Configuration

To send recommendations via email, you'll need to configure Gmail App Passwords:

1. Enable **2-Step Verification** on your Google account.
2. Create an **App Password** for your project.
3. Replace the `sender_password` in the code with your generated App Password.

## Model Training

The **Random Forest Classifier** is trained on the following features from the place data:
- **Rating**: The average rating of the place.
- **Number of Reviews**: The total number of reviews.
- **Activity Similarity Score**: Computed using **cosine similarity** between user activities and place activities.
- **Review Scale**: Normalized review count for places.

The model predicts the probability that a place will be recommended to a user based on these features.

## Future Enhancements

1. **Additional Features**: Implementing more user-specific preferences such as budget, travel time, and accommodation options.
2. **Improved Recommendations**: Incorporating user feedback for more accurate future recommendations.
3. **Scalability**: Moving to a more robust database like **PostgreSQL** or **MongoDB** for larger datasets.

