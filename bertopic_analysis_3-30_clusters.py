import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import re
import emoji
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    # Custom stopwords
    custom_stop_words = {
        'foryoupage', 'ig', 'ago', 'tiktok', 'the', 'is', 'in', 'and', 'to', 'of',
        'a', 'with', 'for', 'on', 'that', 'by', 'this', 'it', 'an', 'be', 'as',
        'at', 'from', 'he', 'she', 'it', 'they', 'we', 'you', 'but', 'or', 'yet',
        'so', 'nor', 'above', 'below', 'under', 'over', 'between', 'among', 'am',
        'are', 'was', 'were', 'be', 'been', 'being', 'this', 'that', 'these',
        'those', 'each', 'every', 'some', 'any', 'very', 'really', 'just', 'quite',
        'too', 'also', 'my', 'your', 'yours', 'I', 'com', 'r', 'reddit', 'can',
        'will', 'our', 'up', 'have', 'had', 'w', 'all', 'if', 'her', 'hers',
        'not', 't', 'us', 'its', 'do', 'i', 's', '3', 'fl', 'www', 'out',
        'more', 'what', 'how', 'has', 'about', 'pm', '10', '00', 'm', 'may'
    }

    # Combine custom stopwords with ENGLISH_STOP_WORDS
    stop_words = ENGLISH_STOP_WORDS.union(custom_stop_words)

    # Print combined base stopwords (custom + ENGLISH_STOP_WORDS)
    print("=== STOPWORDS: Custom + ENGLISH_STOP_WORDS ===")
    base_stopwords_sorted = sorted(stop_words)
    print(base_stopwords_sorted)
    print(f"Total base stopwords (custom + ENGLISH_STOP_WORDS): {len(stop_words)}\n")

    # Function to remove emojis
    def remove_emojis(text):
        return emoji.replace_emoji(text, replace="")

    # Clean the posts
    def clean_text(text):
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        # Remove non-word characters and numbers
        text = re.sub(r'\W+', ' ', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize and convert to lowercase
        tokens = text.lower().split()
        # Remove stop words and short words
        tokens = [
            word for word in tokens
            if word not in stop_words
            and len(word) > 1
        ]
        return tokens

    # Load the dataset: the script expects an Excel workbook with a Post column
    print("Loading dataset...")
    file_name = 'input_posts.xlsx'
    df = pd.read_excel(file_name)
    print(f"Dataset loaded. Total posts: {len(df)}")

    # Remove blank or NaN values in the 'Post' column
    print("Removing blank or NaN posts...")
    df = df[df['Post'].notna()]
    df = df[df['Post'].str.strip() != ""]
    print(f"Blank or NaN posts removed. Remaining posts: {len(df)}")

    # Remove emojis from posts
    print("Removing emojis from posts...")
    df['Post'] = df['Post'].apply(remove_emojis)
    print(f"Emojis removed. Total posts remain unchanged: {len(df)}")

    # Remove duplicate posts
    print("Removing duplicate posts...")
    df = df.drop_duplicates(subset='Post')
    print(f"Duplicate posts removed. Remaining posts: {len(df)}")

    # Extract the posts column
    print("Extracting posts...")
    posts = df['Post'].tolist()
    print(f"Posts extracted. Total posts: {len(posts)}")

    # Get CountVectorizer's built-in English stopwords (same config as used later)
    vectorizer_for_list = CountVectorizer(stop_words="english")
    cv_stopwords = set(vectorizer_for_list.get_stop_words())

    print("=== STOPWORDS: CountVectorizer Built-In English Stopwords ===")
    cv_stopwords_sorted = sorted(cv_stopwords)
    print(cv_stopwords_sorted)
    print(f"Total CountVectorizer stopwords: {len(cv_stopwords)}\n")

    # Full combined stopword universe used anywhere in the pipeline
    # (text cleaning + vectorizer)
    full_stopword_set = stop_words.union(cv_stopwords)
    full_stopwords_sorted = sorted(full_stopword_set)

    print("=== FULL STOPWORD SET USED IN PIPELINE ===")
    print(full_stopwords_sorted)
    print(f"Total stopwords in full pipeline (base + CountVectorizer): {len(full_stopword_set)}\n")

    # Clean the posts
    print("Cleaning posts...")
    cleaned_posts = [clean_text(post) for post in posts]
    print(f"Posts cleaned. Total cleaned posts: {len(cleaned_posts)}")

    # Prepare dictionary and corpus once, since these don't depend on nr_topics
    dictionary = Dictionary(cleaned_posts)
    corpus = [dictionary.doc2bow(post) for post in cleaned_posts]

    # We'll store overall coherence for each number of topics
    # Columns: NumTopics, UMass, CV, C_UCI, C_NPMI
    coherence_results = []

    # Run models for number_of_topics from 3 to 30
    for num_topics in range(3, 31):
        print(f"\nRunning BERTopic model for {num_topics} topics...")

        # Initialize BERTopic model
        vectorizer_model = CountVectorizer(stop_words="english")
        topic_model = BERTopic(nr_topics=num_topics, vectorizer_model=vectorizer_model)

        # Fit the model on the cleaned posts
        print("Fitting BERTopic model on posts...")
        topics, probabilities = topic_model.fit_transform([' '.join(post) for post in cleaned_posts])
        print(f"BERTopic model fitted for {num_topics} topics. Total distinct topics: {len(set(topics))}")

        # Add topic labels to the dataframe
        df_current = df.copy()
        df_current['topic'] = topics

        # Calculate Coherence Scores
        print("Calculating coherence scores...")
        topic_words = topic_model.get_topics()
        coherence_scores = {}

        for topic_id, words in topic_words.items():
            if topic_id == -1:  # Skip outlier topic
                continue
            topic_terms = [word for word, _ in words]
            coherence_scores[topic_id] = topic_terms

        # If no valid topics found (other than -1), skip coherence calculations
        if len(coherence_scores) == 0:
            print("No valid topics found, skipping coherence calculation.")
            # Write minimal results
            output_file = f'clustered_posts_with_analyses_{num_topics}Clusters.xlsx'
            with pd.ExcelWriter(output_file) as writer:
                df_current.to_excel(writer, index=False, sheet_name='Clustered Posts')
            continue

        # Compute coherence for each measure
        coherence_model_umass = CoherenceModel(
            topics=[terms for terms in coherence_scores.values()],
            corpus=corpus,
            dictionary=dictionary,
            coherence='u_mass'
        )
        umass_score = coherence_model_umass.get_coherence()

        coherence_df_umass = pd.DataFrame({
            "Topic": list(coherence_scores.keys()),
            "Coherence Score": coherence_model_umass.get_coherence_per_topic()
        })

        coherence_model_cv = CoherenceModel(
            topics=[terms for terms in coherence_scores.values()],
            texts=cleaned_posts,
            dictionary=dictionary,
            coherence='c_v'
        )
        cv_score = coherence_model_cv.get_coherence()
        coherence_df_cv = pd.DataFrame({
            "Topic": list(coherence_scores.keys()),
            "Coherence Score": coherence_model_cv.get_coherence_per_topic()
        })

        coherence_model_cuci = CoherenceModel(
            topics=[terms for terms in coherence_scores.values()],
            texts=cleaned_posts,
            dictionary=dictionary,
            coherence='c_uci'
        )
        cuci_score = coherence_model_cuci.get_coherence()
        coherence_df_cuci = pd.DataFrame({
            "Topic": list(coherence_scores.keys()),
            "Coherence Score": coherence_model_cuci.get_coherence_per_topic()
        })

        coherence_model_cnpmi = CoherenceModel(
            topics=[terms for terms in coherence_scores.values()],
            texts=cleaned_posts,
            dictionary=dictionary,
            coherence='c_npmi'
        )
        cnpmi_score = coherence_model_cnpmi.get_coherence()
        coherence_df_cnpmi = pd.DataFrame({
            "Topic": list(coherence_scores.keys()),
            "Coherence Score": coherence_model_cnpmi.get_coherence_per_topic()
        })

        # Append to coherence_results
        coherence_results.append({
            "NumTopics": num_topics,
            "U_Mass": umass_score,
            "C_V": cv_score,
            "C_UCI": cuci_score,
            "C_NPMI": cnpmi_score
        })

        # Analyze the most common words in each topic
        print("Analyzing topics for common words...")
        common_words_dict = {}
        unique_topics = set(topics)
        if -1 in unique_topics:
            unique_topics.remove(-1)
        for topic in unique_topics:
            topic_posts = df_current[df_current['topic'] == topic]['Post']
            all_words_topic = []
            for post in topic_posts:
                all_words_topic.extend(clean_text(post))
            common_words_topic = Counter(all_words_topic).most_common(10)
            common_words_dict[topic] = common_words_topic
            print(f"Topic {topic}: {common_words_topic}")

        # Calculate topic sizes
        print("Calculating topic sizes...")
        topic_sizes = df_current['topic'].value_counts().sort_index()
        print("Topic sizes calculated.")

        # Save all results to a single Excel file
        output_file = f'clustered_posts_with_analyses_{num_topics}Clusters.xlsx'
        print(f"Saving analyses to {output_file}...")

        with pd.ExcelWriter(output_file) as writer:
            # Save main dataframe with topics
            df_current.to_excel(writer, index=False, sheet_name='Clustered Posts')

            # Save topic sizes
            topic_sizes.to_excel(writer, sheet_name='Topic Sizes', header=["Size"])

            # Save common words for each topic
            if common_words_dict:
                common_words_df = pd.DataFrame.from_dict(common_words_dict, orient='index')
                common_words_df.columns = [f'Word {i+1}' for i in range(common_words_df.shape[1])]
                common_words_df.to_excel(writer, sheet_name='Common Words')

            # Save coherence scores (one sheet per metric)
            coherence_df_umass.to_excel(writer, sheet_name='Coherence UMass', index=False)
            coherence_df_cv.to_excel(writer, sheet_name='Coherence CV', index=False)
            coherence_df_cuci.to_excel(writer, sheet_name='Coherence C_UCI', index=False)
            coherence_df_cnpmi.to_excel(writer, sheet_name='Coherence C_NPMI', index=False)

            # Save stopwords: base set and full pipeline set
            base_stopwords_df = pd.DataFrame(base_stopwords_sorted, columns=['Stopword'])
            base_stopwords_df.to_excel(writer, sheet_name='Stopwords_Base', index=False)

            all_stopwords_df = pd.DataFrame(full_stopwords_sorted, columns=['Stopword'])
            all_stopwords_df.to_excel(writer, sheet_name='Stopwords_All', index=False)

        print(f"Finished saving for {num_topics} topics.")

    # After all models are run, create a summary plot of overall coherence vs number of topics
    coherence_summary_df = pd.DataFrame(coherence_results)
    coherence_summary_df = coherence_summary_df.sort_values(by="NumTopics")

    # Create output directory for plots if it doesn't exist
    plots_dir = "coherence_plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Plot each metric separately
    metrics = ["U_Mass", "C_V", "C_UCI", "C_NPMI"]
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        plt.plot(coherence_summary_df["NumTopics"], coherence_summary_df[metric], marker='o')
        plt.title(f"{metric} Coherence vs. Number of Topics")
        plt.xlabel("Number of Topics")
        plt.ylabel(f"{metric} Coherence Score")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{metric}_coherence_vs_topics.png"))
        plt.close()

    # Save the overall coherence summary DataFrame to Excel
    coherence_summary_df.to_excel("Overall_Coherence_Summary.xlsx", index=False)
    print("Overall coherence summary and plots saved.")
