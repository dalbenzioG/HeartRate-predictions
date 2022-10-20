import csv
import torch
from matplotlib import pyplot as plt

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np



class AverageMeter(object) :
    """Computes and stores the average and current value"""

    def __init__(self) :
        self.reset()

    def reset(self) :
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self , val , n=1) :
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object) :

    def __init__(self , path , header) :
        self.log_file = open(path , 'w')
        self.logger = csv.writer(self.log_file , delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self) :
        self.log_file.close()

    def log(self , values) :
        write_values = []
        for col in self.header :
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path) :
    with open(file_path , 'r') as input_file :
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs , targets) :
    batch_size = targets.size(0)

    _ , pred = outputs.topk(1 , 1 , True)
    pred = pred.t()
    correct = pred.eq(targets.view(1 , -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size

class Optimization :
    def __init__(self , model , loss_fn , optimizer) :
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self , x , y) :
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x.to(device)).to(device)

        # Computes loss
        loss = self.loss_fn(y.unsqueeze(1).to(device) , yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self , train_loader , val_loader , batch_size=64 , n_epochs=50) :
        model_path = './models/'

        for epoch in range(1 , n_epochs + 1) :
            batch_losses = []
            for x_batch , y_batch in train_loader :
                x_batch = x_batch.to(device)
                x_batch = x_batch.to(torch.float32)
                y_batch = y_batch.to(device).to(torch.float32)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad() :
                batch_val_losses = []
                for x_val , y_val in val_loader :
                    x_val = x_val.to(device)
                    x_val = x_val.to(torch.float32)
                    y_val = y_val.to(device).to(torch.float32)
                    self.model.eval()
                    yhat = self.model(x_val).to(device)
                    val_loss = self.loss_fn(y_val.unsqueeze(1) , yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0) :
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        torch.save(self.model.state_dict() , model_path + 'checkpoint.tar')

    def evaluate(self , test_loader , batch_size=1 , n_features=1) :
        with torch.no_grad() :
            predictions = []
            values = []
            for x_test , y_test in test_loader :
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test).to(device)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())

        return predictions , values

    def plot_losses(self) :
        plt.plot(self.train_losses , label="Training loss")
        plt.plot(self.val_losses , label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()


