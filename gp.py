import torch
import gpytorch
import numpy as np



# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        

class GP():

    def __init__(self, true_function, x_in, y_in, budget, spacing, noise_scaling, initialize, input_form='values'):
        self.model = None
        self.likelihood = None
        self.train_x = None
        self.train_y = None
        self.true_function = true_function
        self.x_in = x_in
        self.y_in = y_in
        self.budget = budget
        self.spacing = spacing
        self.noise_scaling = noise_scaling

        if initialize==True:
            self.initialize_model(input_form)

    def initialize_model(self, input_form):

        if input_form == 'function':
            train_x = torch.linspace(0, int(self.budget), self.spacing)
            self.train_y = self.true_function(self.train_x) + torch.randn(self.train_x.size()) * self.noise_scaling
            # self.train_y = self.true_function(self.train_x)
        elif input_form == 'values':
            train_x = np.array(self.x_in[::self.spacing], dtype=np.float32)
            self.train_y = torch.from_numpy(np.array(self.y_in[::self.spacing], dtype=np.float32) + np.randn(train_x.shape)*self.noise_scaling)
            # self.train_y = torch.from_numpy(np.array(self.y_in[::self.spacing], dtype=np.float32)) + torch.randn(self.train_x.size()) * self.noise_scaling
            self.train_x = torch.from_numpy(train_x)

            for i in range(self.train_y.shape[0]):
                self.train_y[i] = max(self.train_y[i], .01)
                self.train_y[i] = min(self.train_y[i], .99)
        else:
            raise ValueError(input_form)
        

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood)

        self.model.train()
        self.likelihood.train()


        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=0.1)

        training_iter = 50
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f0   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                self.model.covar_module.base_kernel.lengthscale.item(),
                self.model.likelihood.noise.item()
            ))
            optimizer.step()

    def get_best_visit_time(self, min_time, max_time, discretization):
        possible_times = torch.linspace(float(min_time), float(max_time), max(2, discretization))

        if self.model is None:
            return random.choice(list(possible_times.numpy()))

        elif (self.train_x.shape[0] < 2):
            return random.choice(list(possible_times.numpy()))

        elif (self.train_x.shape[0] == 2) and (self.train_x[0] == self.train_x[1]):
            return random.choice(list(possible_times.numpy()))

        else:
            self.model.eval()
            self.likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():

                observed_pred = self.likelihood(self.model(possible_times))
                lower, upper = observed_pred.confidence_region()
                diff = upper.numpy() - lower.numpy()

                best_indices = np.argsort(diff)[::-1]

                return possible_times.numpy()[best_indices[0]]


    def get_best_n_visit_times(self, min_time, max_time, discretization, n):
        possible_times = torch.linspace(min_time, max_time, max(1, discretization))

        if self.model is None:
            times = possible_times.numpy()
            np.random.shuffle(times)
            times = list(times)
            return times[0:n]

        elif (self.train_x.shape[0] < 2):
            times = possible_times.numpy()
            np.random.shuffle(times)
            times = list(times)
            return times[0:n]

        elif (self.train_x.shape[0] == 2) and (self.train_x[0] == self.train_x[1]):
            times = possible_times.numpy()
            np.random.shuffle(times)
            times = list(times)
            return times[0:n]

        else:
            self.model.eval()
            self.likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():

                observed_pred = self.likelihood(self.model(possible_times))
                lower, upper = observed_pred.confidence_region()
                diff = upper.numpy() - lower.numpy()

                best_indices = np.argsort(diff)[::-1]

                best_times = []
                indices = best_indices[0:n]
                for index in indices:
                    best_times.append(possible_times.numpy()[index])
                return best_times


    def get_prediction(self, visit_time):
        max_value = 1000.0

        if self.model is None:
            score = max_value

        elif (self.train_x.shape[0] < 2):
            score = max_value

        elif (self.train_x.shape[0] == 2) and (self.train_x[0] == self.train_x[1]):
            score = max_value
        
        else:
            self.model.eval()
            self.likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.as_tensor(np.array([visit_time]), dtype=torch.float32)

                observed_pred = self.likelihood(self.model(test_x))
                pred = observed_pred.mean.numpy()

                # pred = self.model(test_x)

        return pred



    def get_score(self, vist_time):
        max_value = 1000.0

        if self.model is None:
            score = max_value

        elif (self.train_x.shape[0] < 2):
            score = max_value

        elif (self.train_x.shape[0] == 2) and (self.train_x[0] == self.train_x[1]):
            score = max_value
        
        else:
            self.model.eval()
            self.likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.as_tensor(np.array([vist_time]), dtype=torch.float32)

                observed_pred = self.likelihood(self.model(test_x))
                lower, upper = observed_pred.confidence_region()
                score = (abs(upper.numpy()[0] - lower.numpy()[0]))

        return score


    def calculate_total_uncertainty(self):

        max_value = 100000.0
        
        if self.model is None:
            return max_value

        elif (self.train_x.shape[0] < 2):
            return max_value

        elif (self.train_x.shape[0] == 2) and (self.train_x[0] == self.train_x[1]):
            score = max_value

        else:
            self.model.eval()
            self.likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.linspace(0, int(self.budget), int(self.budget))
                observed_pred = self.likelihood(self.model(test_x))

                lower, upper = observed_pred.confidence_region()
                diff = upper.numpy() - lower.numpy()
                return np.sum(diff)


    def calculate_absolute_error(self):

        max_value = 100000.0
        
        if self.model is None:
            return max_value

        elif (self.train_x.shape[0] < 2):
            return max_value

        elif (self.train_x.shape[0] == 2) and (self.train_x[0] == self.train_x[1]):
            score = max_value

        else:
            self.model.eval()
            self.likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.linspace(0, int(self.budget), int(self.budget))
                observed_pred = self.likelihood(self.model(test_x))

                test_y = self.true_function(test_x)

                diff = np.absolute(test_y.numpy() - observed_pred.mean.numpy())
                return np.sum(diff)


    def visualize(self):

        if self.model is None:
            return

        elif (self.train_x.shape[0] < 2):
            return

        elif (self.train_x.shape[0] == 2) and (self.train_x[0] == self.train_x[1]):
            return

        else:
            self.model.eval()
            self.likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.linspace(0, int(self.budget), int(self.budget))
                observed_pred = self.likelihood(self.model(test_x))

                test_y = self.true_function(test_x)

            with torch.no_grad():
                # Initialize plot
                f, ax = plt.subplots(1, 1, figsize=(4, 3))

                # Get upper and lower confidence bounds
                lower, upper = observed_pred.confidence_region()
                # Plot training data as black stars
                ax.plot(self.train_x.numpy(), self.train_y.numpy(), 'k*')
                # ax.plot(test_x.numpy(), test_y.numpy(), 'k*')
                # Plot predictive means as blue line
                ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
                ax.plot(test_x.numpy(), test_y.numpy(), 'r')
                # Shade between the lower and upper confidence bounds
                ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
                ax.set_ylim([-3, 3])
                ax.legend(['Observed Data', 'Mean', 'Confidence'])

                plt.show()
            return


    def update_model(self, curr_time):
        y = generate_measurement(self.true_function, curr_time)

        if self.train_x is None:
            self.train_x = torch.as_tensor(np.array([curr_time]), dtype=torch.float32)
        else:
            self.train_x = torch.cat((self.train_x, torch.as_tensor(np.array([curr_time]), dtype=torch.float32)))

        if self.train_y is None:
            self.train_y = torch.as_tensor(np.array([y]), dtype=torch.float32)
        else:
            self.train_y = torch.cat((self.train_y, torch.as_tensor(np.array([y]), dtype=torch.float32)))

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood)

        self.model.train()
        self.likelihood.train()


        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=0.1)

        training_iter = 50
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f0   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                self.model.covar_module.base_kernel.lengthscale.item(),
                self.model.likelihood.noise.item()
            ))
            optimizer.step()