import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# generate cube
xs = torch.linspace(-1, 1, steps=10)
ys = torch.linspace(-1, 1, steps=10)

# suface points
x_face = torch.cat([
  xs,
  torch.ones_like(ys),
  xs, 
  -torch.ones_like(ys)
])
x_face_noise = x_face + torch.randn_like(x_face) * 0.05
y_face = torch.cat([
  torch.ones_like(ys),
  ys,
  -torch.ones_like(ys),
  ys
])
y_face_noise = y_face + torch.randn_like(x_face) * 0.05

# outer points
x_out = x_face * 1.5
y_out = y_face * 1.5

# inner points
x_in = torch.zeros(1)
y_in = torch.zeros(1)

# training data
x_train = torch.cat([
  x_face_noise, x_out, x_in
], dim=-1)
y_train = torch.cat([
  y_face_noise, y_out, y_in
], dim=-1)
pos_train = torch.stack([x_train, y_train], dim=-1)
z_train = torch.cat([
  torch.zeros_like(x_face_noise),
  torch.ones_like(x_out),
  -torch.ones_like(x_in)
], dim=-1)

class ThinPlateRegularizer(gpytorch.kernels.Kernel):
  is_stationary = True

  def __init__(self, dist_prior=None, dist_constraint=None, **kwargs):
    super().__init__(**kwargs)
    self.register_parameter(
      name="max_dist",
      parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
    )
    if dist_constraint is None:
      dist_constraint = gpytorch.constraints.GreaterThan(0.20)
    self.register_constraint("max_dist", dist_constraint)
    if dist_prior is not None:
      self.register_prior(
        "dist_prior",
        dist_prior,
        lambda m: m.length,
        lambda m, v: m._set_length(v),
      )

    @property
    def maxdist(self):
      return self.raw_dist_constraint.transform(self.max_dist)

    @maxdist.setter
    def maxdist(self, value):
      return self._set_maxdist(value)

    def _set_maxdist(self, value):
      if not torch.is_tensor(value):
        value = torch.as_tensor(value).to(self.max_dist)
      self.initialize(max_dist=self.raw_dist_constraint.inverse_transform(value))

    def forward(self, x1, x2, **params):
      diff = self.covar_dist(x1, x2, **params)
      diff.where(diff == 0, torch.as_tensor(1e-20))
      noise = 1e-5
      white = noise * torch.eye(diff.shape[0], diff.shape[1])
      tp = (
        2 * torch.pow(diff, 3)
        - 3 * self.max_dist * torch.pow(diff, 2)
        + self.max_dist**3
      )
      return white + tp

class thinPlateModel(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood):
    super(thinPlateModel, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ZeroMean()
    self.covar_module = ThinPlateRegularizer()

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = thinPlateModel(pos_train, z_train, likelihood)

hypers = {
  'likelihood.noise_covar.noise': torch.tensor(0.05),
  'covar_module.max_dist': torch.tensor(0.5),
}
model_params = model.initialize(**hypers)

training_iter = 30
# Find optimal model hyperparameters
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
  # Zero gradients from previous iteration
  optimizer.zero_grad()
  # Output from model
  output = model(pos_train)
  # Calc loss and backprop gradients
  output.log_prob(z_train)
  loss = -mll(output, z_train)
  loss.backward()
  print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
    i + 1, training_iter, loss.item(),
    model.covar_module.base_kernel.lengthscale.item(),
    model.likelihood.noise.item()
  ))
  optimizer.step()
