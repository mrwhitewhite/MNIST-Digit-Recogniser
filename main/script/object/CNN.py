import torch.nn as nn

class CNN(nn.Module):
    def __init__(
        self,
        C: int,
        dropout: float,
        channel_factor: int = 2,
        pool_kernal: int = 2,
    ) -> None:
        """input shape = (B, C, H, W)

        Args:
            C (int): channel size
            dropout (float, optional): dropout probability. Defaults to 0.5.
            channel_factor (int, optional): channel increase by this factor. Defaults to 2.
            pool_kernal (int, optional): pool kernal size. Defaults to 2.

        input shape = (B, C, H, W) -> output shape = (B, C*channel_factor, H/pool_kernal, W/pool_kernal)
        """
        super().__init__()

        self.sequential = nn.Sequential(
            # input shape = (B, C, H, W)
            nn.Conv2d(C, C * channel_factor, 3, padding=1),  # output shape = (B, C*channel_factor, H, W)
            nn.BatchNorm2d(C * channel_factor),  # output shape = (B, C*channel_factor, H, W)
            nn.ReLU(),  # output shape = (B, C*channel_factor, H, W)
            nn.MaxPool2d(pool_kernal),  # output shape = (B, C*channel_factor, H/pool_kernal, W/pool_kernal)
            nn.Dropout(dropout),  # output shape = (B, C*channel_factor, H/pool_kernal, W/pool_kernal)
        )

    def forward(self, x):
        """forward pass

        Args:
            x (torch.Tensor): input tensor with shape (B, C, H, W)

        Returns:
            torch.Tensor: output tensor with shape (B, C*channel_factor, H/pool_kernal, W/pool_kernal)
        """
        return self.sequential(x)