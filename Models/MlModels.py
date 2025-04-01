import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Importing sklearn Modules for training Random Forest Regressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class MlModels:
    def normalize_cosine_similarity(self, cosine_sim):
        """
        Normalize the cosine similarity value to fall between 0 and 1.
        Args:
            cosine_sim: raw cosine similarity value
        Returns:
            Normalized cosine similarity
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(np.array(cosine_sim).reshape(-1, 1))[0][0]

    def load_sbert_model(self):
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        return sbert_model

    def load_TfIdfVectorizer(self, resume_tokens, jd_tokens):
        """
            TF-IDF Vectorizer for pre-tokenized text
            Args:
                resume_tokens: List of tokens from resume (e.g., ["python", "machine", "learning"])
                jd_tokens: List of tokens from job description
            Returns:
                Dictionary of {word: IDF_weight}
        TF-IDF (Term Frequency-Inverse Document Frequency) assigns importance scores to words in a document based on their frequency in a document and across multiple documents.
        * Key Insights from TF-IDF:
        1. Feature Importance: Identifies which words contribute the most to document uniqueness.
        2. Filtering Out Common Words: Reduces the impact of generic words that appear across all resumes (e.g., "team", "work", "experience").00
        3. Skill Emphasis: Highlights relevant skills by giving them higher importance in the vector space.
        4. Sparse Representation: Provides numerical representations useful for model inputs and interpretability.
        """
        # Convert tokens to space-separated strings
        resume_text = ' '.join(resume_tokens)
        jd_text = ' '.join(jd_tokens)

        # Configure TF-IDF with resume-specific settings
        vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
        min_df=1,
        max_df=0.85,
        tokenizer=lambda x: x.split(),
        preprocessor = None,
        lowercase = False
        )

        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        feature_names = vectorizer.get_feature_names_out()
        idf_values = vectorizer.idf_

        # Create dictionary of word to IDF weight
        word_weights = {word: idf for word, idf in zip(feature_names, idf_values)}
        return word_weights

    def get_weighted_embeddings_from_sbert(self, tokens, word_weights, sbert_model=None):
        word_embeddings = []
        weights = []

        if sbert_model is None:
            sbert_model = self.load_sbert_model()

        for token in tokens:
            token_embedding = sbert_model.encode([token], show_progress_bar = False)[0]
            word_embeddings.append(token_embedding)

            weight = word_weights.get(token.lower(), np.median(list(word_weights.values())))
            weights.append(weight)

        # Convert to numpy arrays
        word_embeddings = np.array(word_embeddings)
        weights = np.array(weights)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Calculate weighted average
        weighted_embedding = np.sum(word_embeddings * weights.reshape(-1, 1), axis=0)

        return weighted_embedding

    def train_scoring_model(self, resume_embeddings, jd_embeddings, scores = None):
        """
        Train a scoring model using embeddings with sklearn's cosine_similarity
        Args:
            resume_embeddings: List of resume embeddings (n_samples x embedding_dim)
            jd_embeddings: List of corresponding JD embeddings (n_samples x embedding_dim)
            scores: Optional list of human-rated scores (0-100)
        Returns:
            Trained RandomForestRegressor
        """
        x = []

        resume_embeddings = np.atleast_2d(resume_embeddings)
        jd_embeddings = np.atleast_2d(jd_embeddings)

        # Calculate cosine similarities (returns n_samples x n_samples matrix)
        all_cosine_sims  = cosine_similarity(resume_embeddings, jd_embeddings).diagonal()

        for resume_embedding, jd_embedding, cosine_sims in zip(resume_embeddings, jd_embeddings, all_cosine_sims ):
            # Ensure embeddings are 1D for these operations
            resume_embedding = np.squeeze(resume_embedding)
            jd_embedding = np.squeeze(jd_embedding)

            euclidian_dist = np.linalg.norm(resume_embedding - jd_embedding)
            skill_overlap = np.sum(resume_embedding * jd_embedding)

            x.append([cosine_sims,euclidian_dist,skill_overlap])

        x = np.array(x)

        if scores is None:
            scores = 80 + 20 * all_cosine_sims
            scores = np.atleast_1d(scores)

        # Normalize X features
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        # Verify shapes
        print(f"X shape: {x.shape}")
        print(f"Scores shape: {scores.shape}")

        X_train, X_test, y_train, y_test = train_test_split(x, scores, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(
            n_estimators = 150,
            max_depth = 10,
            min_samples_split = 5,
            random_state = 42
        )

        rf.fit(X_train, y_train)

        #Predictions:
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)

        #Compute Metrics:
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        train_r2 = rf.score(X_train, y_train)
        test_r2 = rf.score(X_test, y_test)

        print(f'Train MSE: {train_mse}')
        print(f'Test MSE: {test_mse}')
        print(f'Train R2: {train_r2}')
        print(f'Test R2: {test_r2}')

        return rf

    def predict_score(self, resume_embeddings, jd_embeddings, rf_model):
        # rf_model = self.train_scoring_model(resume_embeddings, jd_embeddings)
        # Apply StandardScaler to the embeddings (normalize features)
        scaler = StandardScaler()

        resume_embeddings_scaled = scaler.transform(resume_embeddings)
        jd_embeddings_scaled = scaler.transform(jd_embeddings)

        # Create features
        # cosine_sim = np.dot(resume_embeddings, jd_embeddings) / (np.linalg.norm(resume_embeddings) * np.linalg.norm(jd_embeddings))
        cosine_sim = cosine_similarity(resume_embeddings_scaled, jd_embeddings_scaled)[0][0]

        # Normalize cosine similarity
        cosine_sim = self.normalize_cosine_similarity(cosine_sim)

        euclidean_dist = np.linalg.norm(resume_embeddings - jd_embeddings)
        skill_overlap = np.sum(resume_embeddings * jd_embeddings)

        print(rf_model.feature_importances_)
        print(f'Cosine Sim: {cosine_sim}, Euclidian Distance: {euclidean_dist}, Skill Overlap: {skill_overlap}')
        return rf_model.predict([[cosine_sim, euclidean_dist, skill_overlap]])[0]