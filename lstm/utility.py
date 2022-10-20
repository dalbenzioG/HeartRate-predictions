from datetime import datetime
import numpy as np
import torch
from matplotlib import pyplot as plt
from model import RNNModel , LSTMModel , GRUModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model , model_params) :
    models = {
        "rnn" : RNNModel ,
        "lstm" : LSTMModel ,
        "gru" : GRUModel ,
    }
    return models.get(model.lower())(**model_params)


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
        loss = self.loss_fn(y.to(device) , yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self , train_loader , val_loader , batch_size=64 , n_epochs=50 , n_features=1) :
        model_path = './models/'

        for epoch in range(1 , n_epochs + 1) :
            batch_losses = []
            for x_batch , y_batch in train_loader :
                x_batch = x_batch.view([batch_size , -1 , n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch , y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad() :
                batch_val_losses = []
                for x_val , y_val in val_loader :
                    x_val = x_val.view([batch_size , -1 , n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val).to(device)
                    val_loss = self.loss_fn(y_val , yhat).item()
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
                x_test = x_test.view([batch_size , -1 , n_features]).to(device)
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


