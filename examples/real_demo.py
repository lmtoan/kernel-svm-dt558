def filter_pair(label_pair, x, y):
    """ Filter a multi-class dataset into binary dataset given a label pair.
    """
    mask = np.isin(y, label_pair)
    x_bin, y_bin = x[mask].copy(), y[mask].copy()
    y_bin[y_bin==label_pair[0]] = 1.0
    y_bin[y_bin==label_pair[1]] = -1.0
    return x_bin, y_bin

def evaluate(beta, X_train, X_test, y_test, kernel, **kwargs):
    n_test = len(y_test)
    y_pred = np.zeros(n_test)
    y_vals = np.zeros(n_test)
    for i in range(n_test):
        y_vals[i] = np.dot(kernel(X_train, X_test[i, :].reshape(1, -1), **kwargs).reshape(-1), beta)
    y_pred = np.sign(y_vals)
    return np.mean(y_pred != y_test), y_vals  # return error and values from before applying cutoff

def train_predict(X_train, y_train, X_test, y_test, lam, method='ovo', **config):
    error = None
    label_list = np.unique(y_train)
    if method == 'ovo':
        pred_list = []
        label_pair_list = list(itertools.combinations(label_list, 2))
        for label_pair in tqdm(label_pair_list):
            X_train_bin, y_train_bin = filter_pair(label_pair, X_train, y_train)
            beta_vals, train_cache = mylinearsvm(X_train_bin, y_train_bin, lam, **config)
            if config['plot']:
                plt.show(train_cache['plot'])
            _, scores = predict(beta_vals[-1, :], X_train_bin, X_test, gram, **train_cache)
            y_pred_bin = np.zeros_like(y_test) + label_pair[-1]
            y_pred_bin[scores >= 0] = label_pair[0]
            pred_list.append(y_pred_bin)
        test_preds = np.array([mode(pi).mode[0] for pi in np.array(pred_list, dtype=np.int64).T])
        error = np.mean(test_preds != y_test)
    elif method == 'ovr':
        score_list = []
        for label in tqdm(label_list):
            y_train_bin = np.zeros_like(y_train) - 1
            y_train_bin[y_train == label] = 1
            beta_vals, train_cache = mylinearsvm(X_train, y_train_bin, lam, **config)
            if config['plot']:
                plt.show(train_cache['plot'])
            scores = predict(beta_vals[-1, :], X_train, X_test, gram, **train_cache)
            score_list.append(scores)
        test_preds = np.argmax(np.stack(score_list, axis=1), axis=1)
        error = np.mean(test_preds != y_test)
    else:
        print("Method Not Implemented")
    return error, beta_vals