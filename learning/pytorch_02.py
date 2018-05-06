import torch

class Torch_ANN(object):
    def forward(self, inputs, inputWeights, outputWeights):
        h = inputs.mm(inputWeights)
        h_relu = h.clamp(min=0)
        y_pred =  h_relu.mm(outputWeights)
        return h, h_relu, y_pred
    
    def loss(self, y_pred, outputs):
        return (y_pred - outputs).pow(2).sum().item()
    
    def backprop(inputs, outputs, inputWeights, outputWeights, y_pred, h_relu, h):
        grad_y_pred = 2.0 * (y_pred - outputs)
        grad_ow = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(outputWeights.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_iw = inputWeights.t().mm(grad_h)
        return grad_ow, grad_iw
    
    def fit(self, epochs, inputs, outputs, inputWeights, outputWeights, learning_rate):
        
        for t in range(epochs):
            h, h_relu, y_pred = self.forward(inputs, inputWeights, outputWeights)
            loss = self.loss(y_pred, outputs)
            print(t, loss)
            
            grad_ow, grad_iw = self.backprop(inputs, outputs, inputWeights, outputWeights, y_pred, h_relu, h)

if __name__ == '__main__':
    dtype = torch.float
    device = torch.device("cuda:0")
    
    N, D_in, H, D_out = 64, 1000, 100, 10
    
    inputs = torch.randn(N, D_in, device=device, dtype=dtype)
    outputs = torch.randn(N, D_out, device=device, dtype=dtype)
    
    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype)
    
    learning_rate = 1e-6
    
    model = Torch_ANN()
    
    model.fit(500,inputs,outputs,w1,w2,learning_rate)
    