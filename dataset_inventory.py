from sklearn.model_selection import train_test_split
from pipeline import get_dataset, get_model, get_metrics, residual_plot


class DatasetInventory:
    """A class to encapsulate scikit-learn datasets"""
    names = set()

    def __init__(self, name: str):
        self.name = name
        ds = get_dataset(name)
        self.data, self.targ, self.cols, self.desc = (ds.get(x) for x in (
            'data', 'target', 'feature_names', 'DESCR'))
        if self.name not in DatasetInventory.names:
            DatasetInventory.names.add(self.name)

    @classmethod
    def get_generator(cls, names):
        return (cls(name=x) for x in names)

    def split_data(self, train_size=0.8, test_size=0.2, random_state=0):
        splits = train_test_split(
            self.data, self.targ,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state
        )
        return splits

    @staticmethod
    def linear_model(x_train, x_test, y_train, y_test):
        model = get_model('linear_model', 'LinearRegression')
        fit = model.fit(x_train, y_train)
        predictions = fit.predict(x_test)
        mse, r2 = get_metrics(predictions, y_test)
        residual_plot(fitted=predictions, target=y_test, mse=mse, r2=r2)
