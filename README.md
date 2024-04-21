[Source Link](https://www.kaggle.com/c/digit-recognizer)

# Performance Log On CV2
1. Simple Linear + Relu Stack(243658P) = 95%
2. Simple Linear + Relu Stack + nn.Dropout(243658P) = 94%
3. Simple CNN without activation(4266P) = 94%
4. Simple CNN with ReLU activation(4266P) = 95%
5. Simple CNN with ReLU activation + BN (4290P) = 96%
6. CNN + Channel_factor=2 = 93%
7. CNN + Channel_factor=4(8000+P) = 96%
8. CNN + Channel_factor=8(36266+P) = 97%
9. CNN + Channel_factor=8(407722+P) + extended logit calc = 97%