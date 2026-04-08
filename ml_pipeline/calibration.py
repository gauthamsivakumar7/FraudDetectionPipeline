from sklearn.calibration import CalibratedClassifierCV


def calibrate_model(base_estimator, X_train, y_train, method="isotonic", cv=3):
    calibrated = CalibratedClassifierCV(base_estimator, method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated