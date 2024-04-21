class Data:
    """attributes:
    - Train (pd.DataFrame): original train data
    - Test (pd.DataFrame): original test data
    - Submission (pd.DataFrame): original submission data
    - train (pd.DataFrame): train data after splitting
    - cv1 (pd.DataFrame): cv1 data after splitting
    - cv2 (pd.DataFrame): cv2 data after splitting
    - *x_train (torch.Tensor): x data for train
    - *y_train (torch.Tensor): y data for train
    - *x_cv1 (torch.Tensor): x data for cv1
    - *y_cv1 (torch.Tensor): y data for cv1
    - *x_cv2 (torch.Tensor): x data for cv2
    - *y_cv2 (torch.Tensor): y data for cv2
    - *test (torch.Tensor): x data for test
    - nout (int): number of unique outputs
    """

    def __init__(
        self,
        train: str,
        test: str,
        submission: str,
        cv1_size: float = 0.1,
        cv2_size: float = 0.1,
    ) -> None:
        """Data class to load the data and split it into train, test and submission dataframes

        Args:
            train (str): path for train.csv
            test (str): path for test.csv
            submission (str): path for sample_submission.csv
            cv1_size (float, optional): size of cv1 relative to train. Defaults to 0.1.
            cv2_size (float, optional): size of cv2 relative to train. Defaults to 0.1.
        """
        # import original data
        self.Train = pd.read_csv(train)
        self.Test = pd.read_csv(test)
        self.Submission = pd.read_csv(submission)

        # split data into train, cv1, cv2
        self.train, _test = train_test_split(
            self.Train, test_size=cv1_size + cv2_size, random_state=seed, stratify=self.Train["label"]
        )
        self.cv1, self.cv2 = train_test_split(
            _test, test_size=cv2_size / (cv1_size + cv2_size), random_state=seed, stratify=_test["label"]
        )
        del _test

        # split data into X, y
        self.x_train = torch.tensor(self.train.drop(columns=["label"]).values, dtype=torch.float32)
        self.y_train = torch.tensor(self.train["label"].values)
        self.x_cv1 = torch.tensor(self.cv1.drop(columns=["label"]).values, dtype=torch.float32)
        self.y_cv1 = torch.tensor(self.cv1["label"].values)
        self.x_cv2 = torch.tensor(self.cv2.drop(columns=["label"]).values, dtype=torch.float32)
        self.y_cv2 = torch.tensor(self.cv2["label"].values)
        self.test = torch.tensor(self.Test.values, dtype=torch.float32)

        # unique output
        self.nout = len(self.Train["label"].unique())