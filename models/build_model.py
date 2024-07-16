import copy
import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer

def build_optim(args, model, checkpoint, is_bert=False, is_dec=False):
    """ Build optimizer """
    optim_state_key = 'optim' if not is_bert else 'optims'
    lr = args.lr if not is_bert else (args.lr_bert if is_bert else args.lr_dec)
    warmup_steps = args.warmup_steps if not is_bert else (args.warmup_steps_bert if is_bert else args.warmup_steps_dec)

    if checkpoint is not None:
        optim = checkpoint[optim_state_key][0 if is_bert else 1]
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
        optim = Optimizer(
            args.optim, lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=warmup_steps)

    params = [(n, p) for n, p in model.named_parameters() if n.startswith('bert.model')] if is_bert else [(n, p) for n, p in model.named_parameters() if not n.startswith('bert.model')]
    optim.set_parameters(params)
    return optim

def get_generator(vocab_size, dec_hidden_size, device):
    return nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        nn.LogSoftmax(dim=-1)
    ).to(device)

class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained('bert-large-uncased' if large else 'bert-base-uncased', cache_dir=temp_dir)
        self.finetune = finetune

    def forward(self, x, segs, mask):
        if self.finetune:
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec

class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)
        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads, args.ext_dropout, args.ext_layers)

        if args.encoder == 'baseline':
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size, num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if args.max_pos > 512:
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            self._initialize_weights()

        self.to(device)



class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                {n[11:]: p for n, p in bert_from_extractive.items() if n.startswith('bert.model')}, strict=True)

        if args.encoder == 'baseline':
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size, num_hidden_layers=args.enc_layers, num_attention_heads=8, intermediate_size=args.enc_ff_size, hidden_dropout_prob=args.enc_dropout, attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if args.max_pos > 512:
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

