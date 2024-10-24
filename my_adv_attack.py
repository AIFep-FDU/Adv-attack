import torch
import torch.optim as optim
from torch.autograd import Variable


class MyAdvAttack():
    def __init__(self, model, eps=0.031):
        # DO NOT CHANGE THE FOLLOWING
        self.model = model
        self.eps = eps

        # TODO: Add your code from here
        # Example: PGD attack with 10 steps
        self.num_steps = 1                             # 10 Step PGD attack
        self.step_size = eps / self.num_steps           # Set the step size to epsilon / no.steps
        self.ce_fn = torch.nn.CrossEntropyLoss()        # CE loss used by calculate grad

    # TODO: Implement your attack here
    def perturb(self, images, labels):
        # Example of a PGD attack

        # Add random noise and init adversarial example
        X_pgd = images.clone()
        random_noise = torch.FloatTensor(*images.shape).uniform_(-self.eps * 10, self.eps * 10).to(images.device)
        X_pgd = X_pgd + random_noise
        X_pgd = Variable(images.data, requires_grad=True)

        # ===Perturb the input===
        for _ in range(self.num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()                               # Clear grad of adversarial example

            # ===About the model====

            # forward_counter = self.model.get_counter()  # You may access to how many forward pass has been used
            # limit = self.model.get_limit()              # You may access the total forward limit of the evaluation

            # features, logits = self.model.forward_features(X_pgd)     # You may also access to intermediate features
            # features are list of Tensor, outputs from shallow to deep layers
            # length of features may vary.

            # forward_features(), forward(), get_counter(), get_limit() will be provided for all models.

            # ===Perform forward===
            logits = self.model(X_pgd)
            loss = self.ce_fn(logits, labels)
            loss.backward()
            eta = self.step_size * X_pgd.grad.data.sign()               # Calculate perturbation
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)      # Add perturbation

            # Ensure the perturbation within epsilon L-inf norm and between 0 and 1
            eta = torch.nan_to_num(X_pgd.data - images.data, nan=0.0)
            eta = torch.clamp(eta, -self.eps, self.eps)
            X_pgd = Variable(images.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        # ===Output===
        X_pgd = Variable(X_pgd.data, requires_grad=False)               # Your submission should output adv examples
        return X_pgd
