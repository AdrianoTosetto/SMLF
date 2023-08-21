
def get():
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
    import numpy as np
    from sklearn.model_selection import train_test_split
    

    iris = load_iris()
    X = iris['data']
    y = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']

    # One hot encoding
    enc = OneHotEncoder()
    Y = enc.fit_transform(y[:, np.newaxis]).toarray()

    # Scale data to have mean 0 and variance 1 
    # which is importance for convergence of the neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.5, random_state=2)

    n_features = X.shape[1]
    n_classes = Y.shape[1]

    Y_train = Y_train[:, 0].reshape(-1, 1)
    Y_test = Y_test[:, 0].reshape(-1, 1)

    return (X_train, Y_train, X_test, Y_test)
