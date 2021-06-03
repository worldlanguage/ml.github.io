import time

import numpy as np

from sklearn.datasets import load_boston


def read_csv():
    X, y = load_boston(return_X_y=True)

    print(X.shape, y.shape)

    return X, y


def mse_loss(y_true, y_pred):

    return np.sum(np.power(y_true - y_pred, 2)) / y_true.shape[0]


def train_valid_split(X, y, split_rate):
    n_split = int(X.shape[0] * (1 - split_rate))

    return X[:n_split], y[:n_split], X[n_split:], y[n_split:]


lr_params = {
    'learning_rate': 1e-08,
    'n_estimators': 1000,
    'validation_split': 0.2,
    'verbose': 20,
    'seed': 2021,
}


def LinearRegression(learning_rate=1e-8, n_estimators=1000, validation_split=0.2, verbose=20, seed=0):
    lr_params = {
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'validation_split': validation_split,
        'verbose': verbose,
        'seed': seed,
    }

    return lr_params


def fit(lr_params, X, y):
    X_train, y_train, X_valid, y_valid = train_valid_split(X, y, lr_params['validation_split'])

    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, '\n')

    np.random.seed(lr_params['seed'])

    W = np.random.rand(X.shape[1])
    b = np.random.rand()

    loss = -1

    y_pred = np.dot(X_train, W) + b

    print("[Linear Regression] [Training]")

    for i in range(lr_params['n_estimators']):

        dW = np.dot(y_pred - y_train, X_train)
        W = W - lr_params['learning_rate'] * dW

        db = np.sum(y_pred - y_train)
        b = b - lr_params['learning_rate'] * db

        train_mse = mse_loss(np.dot(X_train, W) + b, y_train)
        valid_mse = mse_loss(np.dot(X_valid, W) + b, y_valid)

        y_pred = np.dot(X_train, W) + b

        # time.sleep(0.01)

        if i % lr_params['verbose'] != 0:
            print("\r[{:<4}]   train mse_0's: {:<8.2f}   valid mse_1's: {:<8.2f}".format(i, train_mse, valid_mse),
                  end='')
        else:
            print("\r[{:<4}]   train mse_0's: {:<8.2f}   valid mse_1's: {:<8.2f}".format(i, train_mse, valid_mse),
                  end='\n')

        if loss < 0 or loss * 10 >= valid_mse:
            loss = valid_mse
        else:
            print("\nEarly stopping, best iteration is:")
            print("[{:<4}]   train mse_0's: {:<8.2f}   valid mse_1's: {:<8.2f}".format(i - 1, train_mse, loss),
                  end='\n')
            return None

    return None


if __name__ == '__main__':
    sta_time = time.time()

    X, y = read_csv()

    lr_model = LinearRegression(**lr_params)

    fit(lr_model, X, y)

    print("Time:", time.time() - sta_time)

