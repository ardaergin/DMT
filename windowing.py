import numpy as np
import tensorflow as tf
import datetime


# window generator
class WindowGenerator():
    def __init__(
        self,
        input_width,
        label_width,
        label_column,
        feature_columns,
        df_train,
        df_val,
        df_test,
        num_classes,
        datetime_label,
        regression=False
    ):
        self.regression = regression
        self.feature_columns = feature_columns
        self.num_classes = num_classes
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        # Work out the label column indices.
        self.datetime_label = datetime_label
        self.label_column = label_column
        if not label_column:
            raise ValueError("Label column was not specified")

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width

        self.total_window_size = input_width + label_width

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_column}'])
  
    def split_window(self, df_data) -> tuple[np.ndarray, np.ndarray]:
        """Given a dataframe, splits the dataframe into chunks, where one chunk
        is a number of consecutive days. In case one day is missing in the sequence,
        this sequence is dropped. The result will be X data that is windowed, together
        with Y data that is an array of single prediction values.

        Returns:
            tuple[np.ndarray, np.ndarray]: X_windowed, Y, where X_windowed has shape (num_samples, input_width, amount of features)
            and Y is an array with single values (target values).
        """
        # time column should be day date
        df_data["date"] = df_data[self.datetime_label].dt.date
        df_data = df_data.sort_values("date", ascending=True)

        #used_columns = ["mood","circumplex.arousal","circumplex.valence","activity","screen",
        #"call","sms","appCat_total", "day_sin", "day_cos"]#df_data.select_dtypes(include=['float64', 'int64']).columns
        #sed_columns = used_columns.drop(self.label_column).intersection(pd.Index(self.feature_columns))
        X_windowed = np.zeros(shape=(0, self.input_width, len(self.feature_columns)))
        Y = np.empty(shape=(0,))
        # iterate through subjects
        for subject_id in df_data["id"].unique():
            df_subject = df_data[df_data["id"] == subject_id]
        
            obs_start = df_subject["date"].min()
            obs_end = df_subject["date"].max()
            
            if not isinstance(obs_start, datetime.date):
                continue
            current_window_start = obs_start
            # minus one since we include the first and last point
            current_window_end = current_window_start + datetime.timedelta(days=self.input_width-1)
            
            while current_window_end <= obs_end:
                data_sample = df_subject[
                ((df_subject["date"] >= current_window_start) &\
                (df_subject["date"] <= current_window_end))
                ]
                if len(data_sample) != self.total_window_size:
                    pass
                else:
                    x = np.expand_dims(data_sample[self.feature_columns].values, axis=0)
                    X_windowed = np.concatenate([
                        X_windowed, x
                        ], axis=0)
                    added_outcome = np.array([data_sample.iloc[-1][self.label_column]])
                    Y = np.concatenate([Y, added_outcome], axis=0)
                #print("added frame")
                current_window_end = current_window_end + datetime.timedelta(days=1)
                current_window_start = current_window_start + datetime.timedelta(days=1)
        print(f"Original sample: {len(df_data)}, windowed data points: {X_windowed.shape[0]}")
        if not self.regression:
            # we have a classification probelm
            Y = tf.one_hot(Y-1, self.num_classes)
        return X_windowed, Y
    
    @property
    def train(self):
        X, y = self.split_window(self.df_train)
        assert X.shape[0] == y.shape[0]
        return X, y

    @property
    def val(self):
        return self.split_window(self.df_val)

    @property
    def test(self):
        return self.split_window(self.df_test)