import torch
from torch.optim import Adam

class Optimizer:
    def __init__(self, method, lr, max_grad_norm, beta1=0.9, beta2=0.999, decay_method=None, warmup_steps=0):
        self.optimizer = Adam
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay_method = decay_method
        self.warmup_steps = warmup_steps

    def set_parameters(self, params):
        self.optimizer = self.optimizer(params, lr=self.lr, betas=(self.beta1, self.beta2))

def build_optimizer(args, model, checkpoint=None):
    if checkpoint is not None:
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError("Error: loaded Adam optimizer from existing model but optimizer state is empty")
    else:
        optim = Optimizer(args.optim, args.lr, args.max_grad_norm, beta1=args.beta1, beta2=args.beta2, decay_method='noam', warmup_steps=args.warmup_steps)
    optim.set_parameters(list(model.named_parameters()))
    return optim
