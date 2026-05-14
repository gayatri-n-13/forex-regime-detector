from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

def detect_regimes(data):

    features = data[
        ['returns', 'volatility', 'rsi', 'macd']
    ]

    scaler = StandardScaler()

    X = scaler.fit_transform(features)

    model = GaussianHMM(
        n_components=4,
        covariance_type='diag',
        n_iter=1000
    )

    model.fit(X)

    hidden_states = model.predict(X)

    data['Regime'] = hidden_states

    return data