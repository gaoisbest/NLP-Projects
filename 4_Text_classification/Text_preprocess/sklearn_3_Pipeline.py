
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import fetch_20newsgroups

# for successful download data set
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


text_clf = Pipeline([('tf_idf_counter', TfidfVectorizer()),
                     ('clf', SVC(kernel='linear', probability=True))])

parameters = {
    'tf_idf_counter__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tf_idf_counter__min_df': (1, 2, 3, 4, 5),
    'clf__C': (1.0, 10.0, 100.0),
    'clf__class_weight' : ('balanced', None)
}

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gs_clf = GridSearchCV(estimator=text_clf, param_grid=parameters, n_jobs=1, cv=10, verbose=10)
gs_clf.fit(X=X_train, y=y_train)

print('best score: ', gs_clf.best_score_)
for param_name in sorted(parameters.keys()):
    print('%s: %r' % (param_name, gs_clf.best_params_[param_name]))
