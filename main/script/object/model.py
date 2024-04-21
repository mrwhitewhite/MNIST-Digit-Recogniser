class Model(nn.Module):
    def __init__(self, data: Data, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        torch.manual_seed(seed)

        # data class
        self.d = data
        
        # model
        self.sequential = nn.Sequential(
            # input shape (B,1,28,28)
            CNN(C=1,dropout=0.1,channel_factor=_cf,pool_kernal=2), # output shape = (B, _cf, 14, 14)
            CNN(C=_cf,dropout=0.05,channel_factor=_cf,pool_kernal=2), # output shape = (B, _cf**2, 7, 7)
            nn.Flatten(), # output shape = (B, (_cf**2)*7*7)
            nn.Linear((_cf**2)*7*7,10), # output shape = (B, 10)
        )

    def forward(self, x: torch.Tensor):
        """forward pass of model

        Args:
            x (torch.Tensor): input data x

        Returns:
            torch.Tensor: output logits shape:(-1,10)
        """
        return self.sequential(x.view(-1, 1, 28, 28))

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """calculate loss of model given y_pred and y_true

        Args:
            y_pred (torch.Tensor): prediction Y of model
            y_true (torch.Tensor): actual Y

        Returns:
            torch.Tensor: loss of model
        """
        return F.cross_entropy(y_pred, y_true)

    @torch.no_grad()
    def loss_test(self):
        """calculate loss of model given cv2 data

        Returns:
            float: loss of cv2
        """
        return self.loss(self.forward(d.x_cv2), d.y_cv2).item()

    def summary(self) -> None:
        """print str of number of parameter to be trauned"""
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(f"Number of parameters: {num_params}")

    @torch.no_grad()
    def accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """calculate accuracy of model given x and y data

        Args:
            x (torch.Tensor): x data
            y (torch.Tensor): y data

        Returns:
            float: accuracy
        """
        return (self.forward(x).argmax(1) == y).float().mean().item()

    @torch.no_grad()
    def accuracy_test(self) -> float:
        """calculate accuracy of model given cv2 data

        Returns:
            float: accuracy
        """
        return self.accuracy(self.d.x_cv2, self.d.y_cv2)

    def fit(
        self,
        mode: Literal["train", "cv1", "cv2"],
        epochs: int,
        iteration: int,
        lr: float,
        lr_patience: int = 10,
        lr_gamma: float = 0.1,
        print_interval: int = 1000,
    ) -> None:
        """fit the model

        Args:
            mode ("train"/"cv1"/"cv2"): mode of fitting
            epochs (int): number of batch per iteration
            iteration (int): number of iteration to fit
            lr (float): learning rate
            print_interval (int, optional): print status every <print_interval> iteration. Defaults to 1000.
        """
        # training mode
        self.train()

        # specify data to be used
        if mode == "train":
            X = d.x_train
            Y = d.y_train
        elif mode == "cv1":
            X = d.x_cv1
            Y = d.y_cv1
        elif mode == "cv2":
            X = d.x_cv2
            Y = d.y_cv2

        # total number of element in data
        n = len(X)

        # optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr)
        lr_step = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_gamma, patience=lr_patience)

        for epoch in range(iteration):
            optimizer.zero_grad()

            # batch
            batch_index = torch.randint(0, n, (epochs,))
            batch_x = X[batch_index]
            batch_y = Y[batch_index]

            # loss,backward,optim step
            l = self.loss(self.forward(batch_x), batch_y)
            l.backward()
            optimizer.step()
            lr_step.step(l)

            # print every print_interval
            if ((epoch + 1) % print_interval == 0) or epoch == 0:
                if epoch == 0:
                    self.summary()
                with torch.no_grad():
                    print(f"\n{epoch+1} Iteration")
                    # print(f"Loss_Train          : {self.loss(self.forward(X), Y).item():.4f}")
                    # print(f"Loss_cv2            : {self.loss_test():.4f}")
                    print(f"Accuracy_Train  : {self.accuracy(X, Y):.4f}")
                    print(f"Accuracy_cv2    : {self.accuracy_test():.4f}")