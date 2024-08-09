import streamlit as st
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

@st.cache_resource
def load_model_and_vectorizer():
    with open('svm_model.pkl', 'rb') as model_file:
        decision_tree_model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return decision_tree_model, vectorizer

@st.cache_data
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df['Feedback'] = (df['Positive Feedback'].fillna('') + ' ' + 
                      df['Negative Feedback'].fillna('') + ' ' + 
                      df['Improvement Suggestions'].fillna('') + ' ' + 
                      df['Additional Comments'].fillna(''))
    df['Feedback'] = df['Feedback'].apply(clean_text)
    df['Sentiment'] = (df['Clarity of Explanations'] + df['Knowledge of Subject'] + 
                       df['Engagement'] + df['Punctuality'] + df['Classroom Management'] + 
                       df['Fairness in Grading'] + df['Approachability'] + df['Availability'] + 
                       df['Encouragement'] + df['Material Relevance'] + 
                       df['Assignment Usefulness'] + df['Course Pace']) / 12
    df['Sentiment'] = df['Sentiment'].apply(lambda x: 1 if x >= 3 else 0)
    return df

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower()

def generate_teacher_report(teacher_id, df, model, vectorizer):
    teacher_df = df[df['Teacher ID'] == teacher_id]
    
    if teacher_df.empty:
        return None, f"No data available for Teacher ID: {teacher_id}"
    
    teacher_feedback = teacher_df['Feedback']
    teacher_feedback_tfidf = vectorizer.transform(teacher_feedback)
    predictions = model.predict(teacher_feedback_tfidf)
    
    positive_feedback_count = sum(predictions)
    total_feedback_count = len(predictions)
    positive_feedback_percentage = (positive_feedback_count / total_feedback_count) * 100
    
    # Define performance categories based on positive feedback percentage
    if positive_feedback_percentage >= 80:
        performance_category = "Excellent"
    elif positive_feedback_percentage >= 60:
        performance_category = "Good"
    elif positive_feedback_percentage >= 40:
        performance_category = "Satisfied"
    else:
        performance_category = "Needs Improvement"
    
    return teacher_df, positive_feedback_percentage, predictions, performance_category

def plot_feedback_distribution(predictions):
    fig, ax = plt.subplots()
    sns.countplot(x=predictions, ax=ax, palette='viridis')
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_title('Feedback Sentiment Distribution')
    return fig

def plot_pie_chart(positive_feedback_percentage, total_feedback_count):
    fig, ax = plt.subplots()
    sizes = [positive_feedback_percentage, 100 - positive_feedback_percentage]
    labels = ['Positive Feedback', 'Negative Feedback']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#4CAF50', '#F44336'])
    ax.axis('equal')
    ax.set_title('Feedback Sentiment Proportion')
    return fig

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

def plot_wordcloud(wordcloud):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def get_sample_negative_sentences(df, vectorizer, model, num_samples=5):
    feedbacks = df['Feedback']
    feedback_tfidf = vectorizer.transform(feedbacks)
    predictions = model.predict(feedback_tfidf)
    
    negative_feedbacks = df[predictions == 0]['Feedback']
    return negative_feedbacks.sample(min(num_samples, len(negative_feedbacks))).tolist()

def generate_overall_performance_summary(df, model, vectorizer):
    teacher_ids = df['Teacher ID'].unique()
    performance_data = []

    for teacher_id in teacher_ids:
        teacher_df, positive_feedback_percentage, predictions, performance_category = generate_teacher_report(teacher_id, df, model, vectorizer)
        if teacher_df is not None:
            performance_data.append({
                'Teacher ID': teacher_id,
                'Positive Feedback Percentage': positive_feedback_percentage,
                'Performance Category': performance_category
            })
    
    performance_df = pd.DataFrame(performance_data)
    return performance_df

def plot_performance_bar_chart(performance_df):
    fig, ax = plt.subplots()
    performance_counts = performance_df['Performance Category'].value_counts()
    performance_counts.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Overall Performance of All Teachers')
    ax.set_xlabel('Performance Category')
    ax.set_ylabel('Number of Teachers')
    return fig

# Streamlit app
st.title('Teacher Feedback Report')

# Load the dataset and model
file_path = 'feedback_dataset.csv'  # Update this to your file path
df = load_dataset(file_path)
decision_tree_model, vectorizer = load_model_and_vectorizer()

# Extract unique teacher IDs
teacher_ids = df['Teacher ID'].unique().tolist()

# Dropdown for teacher ID in the sidebar
teacher_id = st.sidebar.selectbox('Select Teacher ID:', teacher_ids)

if teacher_id:
    teacher_df, positive_feedback_percentage, predictions, performance_category = generate_teacher_report(teacher_id, df, decision_tree_model, vectorizer)
    
    if teacher_df is not None:
        # Generate and display the report in the sidebar
        report = (f"Teacher ID: {teacher_id}\n"
                  f"Total Feedback Count: {len(predictions)}\n"
                  f"Positive Feedback Percentage: {positive_feedback_percentage:.2f}%\n"
                  f"Overall Performance: {performance_category}\n")
        st.sidebar.text(report)

# Overall performance for all teachers
st.subheader('Overall Performance of All Teachers')
performance_df = generate_overall_performance_summary(df, decision_tree_model, vectorizer)

# # Display overall performance statistics and bar chart
st.write(f"Total Teachers Analyzed: {len(performance_df)}")
st.write(f"Average Positive Feedback Percentage: {performance_df['Positive Feedback Percentage'].mean():.2f}%")

# # Plot and display the performance bar chart
# performance_bar_chart = plot_performance_bar_chart(performance_df)
# st.pyplot(performance_bar_chart)

if teacher_id and teacher_df is not None:
    # Display all visualizations in a single row
    st.subheader(f'Feedback Analysis of Teacher ID :{teacher_id}')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Plot and display feedback distribution
        st.subheader('Feedback Sentiment Distribution')
        feedback_dist_fig = plot_feedback_distribution(predictions)
        st.pyplot(feedback_dist_fig)
        
    with col2:
        # Plot and display pie chart of feedback proportion
        st.subheader('Feedback Sentiment Proportion')
        pie_chart_fig = plot_pie_chart(positive_feedback_percentage, len(predictions))
        st.pyplot(pie_chart_fig)
    
    with col3:
        # Generate and display word cloud
        st.subheader('Feedback Word Cloud')
        all_feedback_text = ' '.join(teacher_df['Feedback'])
        wordcloud = generate_wordcloud(all_feedback_text)
        wordcloud_fig = plot_wordcloud(wordcloud)
        st.pyplot(wordcloud_fig)

# Text input for sentiment prediction
# st.subheader('Predict Feedback Sentiment')
# user_feedback = st.text_input('Enter feedback text:')

# if user_feedback:
#     cleaned_feedback = clean_text(user_feedback)
#     feedback_tfidf = vectorizer.transform([cleaned_feedback])
#     prediction = decision_tree_model.predict(feedback_tfidf)[0]
#     sentiment = 'Positive' if prediction == 1 else 'Negative'
#     st.write(f'Sentiment: {sentiment}')

# Display sample negative feedbacks
st.subheader('Feedbacks')
sample_negative_sentences = get_sample_negative_sentences(df, vectorizer, decision_tree_model)
for i, feedback in enumerate(sample_negative_sentences, 1):
    st.write(f"{i}. {feedback}")
