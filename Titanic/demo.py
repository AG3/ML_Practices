import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_raw = pd.read_csv("train.csv")
test_raw = pd.read_csv("test.csv")
train_Y = train_raw.Survived

train = train_raw.drop(labels=['Ticket', 'Cabin', 'Embarked', 'PassengerId', 'Name', 'Survived'], axis=1)
test = test_raw.drop(labels=['Ticket', 'Cabin', 'Embarked', 'PassengerId', 'Name'], axis=1)

train = train.replace('male', 1)
train = train.replace('female', 0)

test = test.replace('male', 1)
test = test.replace('female', 0)


# Fill NaN with mean value
train_mean_age = train.Age.mean()
test_mean_age = test.Age.mean()
test_mean_fare = test.Fare.mean()

train.Age = train.Age.fillna(train_mean_age)
test.Age = test.Age.fillna(test_mean_age)
test.Fare = test.Fare.fillna(test_mean_fare)


def RandomForest():
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(train, train_Y)
    res = clf.predict(test)
    return res

def RandomDenseLayers():
    global train
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.utils import to_categorical
    import numpy as np

    train = train.values
    print(train.shape)

    model = Sequential()
    model.add(Dense(32, input_shape=(6,)))
    model.add(Dense(16))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    caty = to_categorical(train_Y)
    model.fit(train, caty, epochs=1000, batch_size=16)
    pred = model.predict(test)

    res = np.argmax(pred, 1)
    return res


res = RandomDenseLayers()
d = {'PassengerId': test_raw.PassengerId, 'Survived': res}
df = pd.DataFrame(data=d)
df.to_csv("./rdm-forest-submission.csv", index=False)
