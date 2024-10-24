# Submission Format
- The **submission (*.zip)** should contain 2 files (**my_adv_attack.py, metadata**). You may add helper files if needed.
- The my_adv_attack.py should have a class named as M**yAdvAttack()**, with **__init__(self, model, eps=0.031)** function. The arguments will be replaced during testing.
You can have white-box access to the weights of the model. You can assume the model will not use gradient masking.
- MyAdvAttack() should also contain a function **perturb(self, images, labels)**, images are input x, and labels are the ground truth, both as Tensor. Your submission should generate and output adversarial examples using images and labels for the model initialized in MyAdvAttack under the L_inf norm of eps.
- The model will have 2 functions that you can perform forward pass. **forward(x)** will return the output of the logits layer (prior to softmax), **forward_features(x)** will return features from intermediate layers in a list and the logits as tuple ([], Tensor). The features are sequentially constructed from shallow to deeper layers.
- Implement your attack in the perturb function. Feels free add other helper functions.
- The server runs with PyTorch1.9.0, Python3.7, CUDA-10.2.


# Submission
- zip **sample_code_submission** and upload to the server (required: **metadata** and **my_adv_attack.py**)
- Otherfiles *evaluation.py, data_loader.py, utils.py* are a sample testing code, you may use them locally.
