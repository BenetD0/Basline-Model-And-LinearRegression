prices = np.array([float(item.price) for item in train])
documents = [item.summary for item in train]

np.random.seed(42)
vectorizer = CountVectorizer(max_features=2000, stop_words='english')
X = vectorizer.fit_transform(documents)

selected_words = vectorizer.get_feature_names_out()
print(f"Number of selected words: {len(selected_words)}")
print("Selected words:", selected_words[1000:1020])

regressor = LinearRegression()
regressor.fit(X, prices)

def natural_language_linear_regression_pricer(item):
    x = vectorizer.transform([item.summary])
    return max(regressor.predict(x)[0], 0)

    evaluate(natural_language_linear_regression_pricer, test)