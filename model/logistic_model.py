from sklearn.linear_model import LogisticRegression

def train_logistic(X_train_scaled, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model
