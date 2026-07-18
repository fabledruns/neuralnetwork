class MSE:
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        
        return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
    
    def backward(self):
        return [2 * (p - t) / len(self.predictions) for p, t in zip(self.predictions, self.targets)]