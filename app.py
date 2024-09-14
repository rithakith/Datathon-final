# Personalized travel recommendations
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pickle
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Function to load datasets
def load_data():
    return pd.read_csv('user_data_version_3_10K_Users.csv'), pd.read_csv('updated_places_with_activities.csv')

user_data, places_data = load_data()

# Function to process user preference
def process_user_preference(preferred_activities):
    return ' '.join(preferred_activities)

def show_message(title, message, success=True):
    with st.expander(title, expanded=True):
        if success:
            st.success(message)
        else:
            st.error(message)
    
def send_email(receiver_email,user_name, recommended_places_list):
    email_content = f"<h2>Hi {user_name} Your Top 5 Recommended Places in Sri Lanka 🌴</h2>"
    for place in recommended_places_list[:5]:
        activities = place['Available Activities'].split()
        unique_activities = ', '.join(sorted(set(activities)))
        email_content += f"<h3>📍 {place['name']}</h3>"
        email_content += f"<p><strong>Rating:</strong> {place['rating']} ⭐</p>"
        email_content += f"<p><strong>Unique Activities:</strong> {place['Activities']}</p>"
        email_content += f"<p><strong>Total Reviews:</strong> {place['user_ratings_total']} reviews</p>"
        email_content += "<hr>"

    sender_email = "visitsrilankadaredevils@gmail.com"
    sender_password = "xcjq qlyd qbwx obqx"
    subject = "Your Personalized Place Recommendations in Sri Lanka"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(email_content, "html"))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(message)
        server.quit()
        st.session_state.email_sent = True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        st.session_state.email_sent = False

# Streamlit app
st.title("Personalized Place Recommendation System")

# User input form
with st.form(key='user_form'):
    st.subheader("Tell Us About Your Travel Preferences")
    name = st.text_input("Name")
    email = st.text_input("Email")
    preferred_activities_input = st.multiselect(
        "What activities are you interested in?",
        options=places_data['Activities'].str.strip('[]').str.replace("'", "").str.split(', ').explode().unique().tolist()
    )
    bucket_list_destinations = st.text_area(
        "What are your bucket list destinations in Sri Lanka? (Separate with commas)"
    )

    submit_button = st.form_submit_button("Get Recommendations")

    if submit_button:
        if not name or not email or not preferred_activities_input:
            st.error("Please fill all fields.")
        else:
            # Convert the list of preferred activities to a string
            preferred_activities_str = ', '.join(preferred_activities_input)

            # Update the last row of the CSV file with new data
            user_data.iloc[-1, user_data.columns.get_loc('Name')] = name
            user_data.iloc[-1, user_data.columns.get_loc('Email')] = email
            user_data.iloc[-1, user_data.columns.get_loc('Preferred Activities')] = preferred_activities_str
            user_data.iloc[-1, user_data.columns.get_loc('Bucket list destinations Sri Lanka')] = bucket_list_destinations

            # Save updated user data to the CSV file
            user_data.to_csv('user_data_version_3_10K_Users.csv', index=False)

            # Clean and prepare the updated user data
            user_data['Preferred Activities Clean'] = user_data['Preferred Activities']\
                .str.strip('[]')\
                .str.replace("'", "", regex=False)\
                .str.replace(',', ' ', regex=False)
            user_data['Bucket List Destinations Clean'] = user_data['Bucket list destinations Sri Lanka']\
                .str.strip('[]')\
                .str.replace("'", "", regex=False)

            # Example input for preferred activities
            user_preference = [process_user_preference(preferred_activities_input)]

            # Create an activity-to-place mapping
            activity_place_mapping = defaultdict(set)
            for idx, row in user_data.iterrows():
                activities = row['Preferred Activities Clean'].split()
                destinations = row['Bucket List Destinations Clean'].split(', ')
                for destination in destinations:
                    for activity in activities:
                        activity_place_mapping[destination.strip()].add(activity.strip())

            # Add 'Available Activities' column to places_data
            places_data['Available Activities'] = places_data['name'].map(
                lambda x: ' '.join(activity_place_mapping.get(x, []))
            )
            places_data['Available Activities'].fillna('', inplace=True)

            # Compute 'Activity Similarity Score'
            activity_tfidf = TfidfVectorizer(stop_words='english')
            place_activity_vectors = activity_tfidf.fit_transform(places_data['Available Activities'])
            user_activity_vector = activity_tfidf.transform(user_preference)
            activity_similarity_scores = cosine_similarity(user_activity_vector, place_activity_vectors).flatten()
            places_data['Activity Similarity Score'] = activity_similarity_scores

            # Prepare the place dataset
            places_data['rating'].fillna(0, inplace=True)
            places_data['user_ratings_total'].fillna(0, inplace=True)
            places_data['review_scale'].fillna(0, inplace=True)
            places_data['user_ratings_total_normalized'] = places_data['user_ratings_total'] / places_data['user_ratings_total'].max()

            # Prepare labels y
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            user_activity_vectors = tfidf_vectorizer.fit_transform(user_data['Preferred Activities Clean'].fillna(''))
            user_similarity_scores = cosine_similarity(tfidf_vectorizer.transform(user_preference), user_activity_vectors).flatten()
            user_data['Similarity Score'] = user_similarity_scores
            top_similar_users = user_data.sort_values(by='Similarity Score', ascending=False).head(15)
            top_destinations = pd.Series(top_similar_users['Bucket List Destinations Clean'].str.split(', ')).explode().value_counts().head(10)
            recommended_places = top_destinations.index.tolist()

            # Create a label for recommendations
            places_data['is_recommended'] = places_data['name'].apply(lambda x: 1 if x in recommended_places else 0)

            # Prepare feature set and train the model
            X = places_data[['rating', 'user_ratings_total_normalized', 'review_scale', 'Activity Similarity Score']]
            y = places_data['is_recommended']
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            place_predictions = model.predict_proba(X)[:, 1]
            places_data['predicted_probability'] = place_predictions
            sorted_places = places_data.sort_values(by='predicted_probability', ascending=False)

            # Ensure diversity in recommendations
            recommended_places_list = []
            added_places = set()
            for activity in preferred_activities_input:
                activity_places = places_data[places_data['Available Activities'].str.contains(activity)]
                activity_places = activity_places.sort_values(by='predicted_probability', ascending=False)
                activity_places = activity_places[~activity_places['name'].isin(added_places)]
                if not activity_places.empty:
                    top_place = activity_places.iloc[0]
                    recommended_places_list.append(top_place)
                    added_places.add(top_place['name'])

            if len(recommended_places_list) < 5:
                additional_places = sorted_places[~sorted_places['name'].isin(added_places)].head(5 - len(recommended_places_list))
                recommended_places_list.extend(additional_places.to_dict('records'))

            # Display recommendations
            st.write(f"### Hi {name}, here are your Top 5 Recommended Places:")
            for place in recommended_places_list[:5]:
                activities = place.get('Available Activities', '')
                if isinstance(activities, str):
                    unique_activities = ', '.join(sorted(set(activities.split())))
                else:
                    unique_activities = 'No activities listed'

                review_count = int(place.get('user_ratings_total', 0))
                probability_percentage = place.get('predicted_probability', 0) * 100
                st.markdown(f"**📍 {place['name']}**")
                st.markdown(f"- **Probability:**  {probability_percentage}%")
                st.markdown(f"- **Rating:** {place['rating']} ⭐")
                st.markdown(f"- **Unique Activities:** {place['Activities']}")
                st.markdown(f"- **Review Count:** {review_count} reviews")
                st.markdown("---")
            send_email(email, name, recommended_places_list)
            if st.session_state.get('email_sent', False):
                show_message("Email Sent Successfully!", f"Your personalized recommendations have been sent to your email.")
                st.session_state.email_sent = False  # Reset the flag

            # Save the trained model
            with open('random_forest_model.pkl', 'wb') as model_file:
                pickle.dump(model, model_file)
