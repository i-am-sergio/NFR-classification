import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.corpus import wordnet as wn, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

class Prediction:
    def __init__(self, num_classes, model):
        self.tfidf_vocab, self.selected_features, self.pca, self.model, self.label_encoder = self._load_models(12, model)
        self.lemmatizer = WordNetLemmatizer()
        self.language_stopwords = stopwords.words('english')
        self.tag_map = defaultdict(lambda: wn.NOUN)
        self.tag_map.update({'J': wn.ADJ, 'V': wn.VERB, 'R': wn.ADV})

    def _load_models(self, num_classes, model):
        model_filename = model+"_model_2.pkl" 
        if num_classes == 12: 
            model_filename = model+"_model_12.pkl"
        elif num_classes == 11:
            model_filename = model+"_model_11.pkl"
        with open("models/"+model_filename, "rb") as f:
            data = pickle.load(f)
            tfidf_vocab = data["tfidf_vocab"]
            selected_features = data["selected_features"]
            pca = data["pca"]
            model = data["model"]
            label_encoder = data["label_encoder"]

        return tfidf_vocab, selected_features, pca, model, label_encoder

    def preprocess_text(self, text):
        vocab = self.tfidf_vocab["vocab"]
        idf_values = self.tfidf_vocab["idf"]
        vocab_lookup = {word: i for i, word in enumerate(vocab)}

        tokens = word_tokenize(text.lower())
        processed_tokens = [
            self.lemmatizer.lemmatize(word, self.tag_map[tag[0]])
            for word, tag in pos_tag(tokens)
            if word.isalpha() and word not in self.language_stopwords
        ]

        bow_vector = np.zeros(len(vocab), dtype=np.int16)
        for token in processed_tokens:
            if token in vocab_lookup:
                bow_vector[vocab_lookup[token]] += 1

        tf_values = bow_vector / np.sum(bow_vector) if np.sum(bow_vector) > 0 else np.zeros(len(vocab))
        tfidf_vector = tf_values * idf_values

        return pd.DataFrame([tfidf_vector], columns=vocab)

    def filter_features(self, tfidf_vector):
        return tfidf_vector[self.selected_features]

    def apply_pca(self, filtered_vector):
        transformed_vector = self.pca.transform(filtered_vector)
        X_new_cols = [f'Comp{index + 1}' for index in range(transformed_vector.shape[1])]
        return pd.DataFrame(data=transformed_vector, columns=X_new_cols)

    def predict_class(self, transformed_vector):
        predicted_class_index = self.model.predict(transformed_vector)[0]
        return self.label_encoder.inverse_transform([predicted_class_index])[0]

    def process_requirement(self, requirement : str):
        tfidf_vector = self.preprocess_text(requirement)
        filtered_vector = self.filter_features(tfidf_vector)
        transformed_vector = self.apply_pca(filtered_vector)
        predicted_class = self.predict_class(transformed_vector)
        return predicted_class
    
    def get_name_of_class(self, predicted_class):
        """
        Returns the full name of the predicted class based on its abbreviation.

        Args:
            predicted_class (str): The abbreviated predicted class (e.g., 'F', 'SC').

        Returns:
            str: The full name of the predicted class, or "Unknown" if not found.
        """
        class_name_mapping = {
            'F': 'Functional Requirement',
            'A': 'Availability',
            'L': 'Legal',
            'LF': 'Look-and-feel',
            'MN': 'Maintainability',
            'O': 'Operability',
            'PE': 'Performance',
            'SC': 'Scalability',
            'SE': 'Security',
            'US': 'Usability',
            'FT': 'Fault Tolerance',
            'PO': 'Portability'
        }

        if predicted_class in class_name_mapping:
            return class_name_mapping[predicted_class]
        else:
            return "Unknown"


if __name__ == '__main__':
    predictor = Prediction(12,"LR")
    while True:
        user_requirement = input("Enter requirement text (or type 'exit' to quit): ")
        if user_requirement.lower() == 'exit':
            break
        predicted_class = predictor.process_requirement(user_requirement)
        print(f"RQ: '{user_requirement}' => Predicted Class: [{predicted_class}: {predictor.get_name_of_class(predicted_class)}]")

# Probar con el siguiente requisito (esperado: SE):
# The Disputes application shall ensure that only authorized users are allowed to logon to the application. 

