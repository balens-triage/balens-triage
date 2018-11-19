import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_message(message):
    return re.sub(r'bug [0-9]+', '', message, flags=re.IGNORECASE | re.MULTILINE | re.M)


class DeveloperCF:
    def __init__(self, storage, developers):
        commit_logs = storage.load_developer_content(tuple(developers))

        profiles = []
        self._developers = []

        for log in commit_logs:
            messages = [clean_message(msg) for msg in log['messages']]
            self._developers.append(log['email'])

            posts = []
            if 'posts' in log:
                posts = [clean_message(post) for post in log['posts']]

            profiles.append(" ".join(messages + posts))

        # TODO
        self._vectorizer = TfidfVectorizer(lowercase=True,
                                           min_df=5,
                                           stop_words='english',
                                           analyzer='word')

        self._mtrx = self.cosine_dist(profiles)

    def cosine_dist(self, profiles):
        transformed = self._vectorizer.fit_transform(profiles).toarray()

        matrix = cosine_similarity(transformed, transformed)

        np.fill_diagonal(matrix, 0)

        return matrix

    def most_similar_dev(self, developer):
        if developer not in self._developers:
            return developer
        else:
            index = self._developers.index(developer)
            row = self._mtrx[index].tolist()
            recommended_index = row.index(max(row))

            return self._developers[recommended_index]
