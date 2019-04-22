from sklearn.preprocessing import MinMaxScaler


def create_dataset(data_frame, target_columns, need_scale):
    dataset = data_frame[target_columns]
    print(dataset.shape)

    if need_scale:
        scaler = MinMaxScaler()
        dataset.loc[:, target_columns] = scaler.fit_transform(dataset[target_columns])

    grouped_by = dataset.groupby('patient_id')

    X = grouped_by.apply(lambda x: x[:-1])
    Y = grouped_by.apply(lambda x: x[1:])

    X_grouped = grouped_by.apply(lambda x: x[:-1].as_matrix())
    Y_grouped = grouped_by.apply(lambda x: x[1:].as_matrix())

    # print(target_columns)
    # print(X.shape)
    # print(Y.shape)
    # delimiter = int(0.8 * X.shape[0])
    # train_X, train_Y = X[:delimiter], Y[:delimiter]
    # test_X, test_Y = X[delimiter:], Y[delimiter:]

    return X, Y, X_grouped, Y_grouped
