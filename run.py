from sys import version
from tokenizers import Tokenizer
import torch
from models.lstm import Seq2Seq
from dataset import NMTDataset
import click
from torch.utils.data import DataLoader
from NMTtokenizers.tokenizer import *
from utils import *
import time
from models.transformer import TransformerModel

CONTEXT_SETTINGS = dict(help_option_names = ['-h', '--help'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
embed_size = 300
hidden_size = 256
dropout_rate = 0.5
n_layers = 1
beam_size = 5
epoch = 30
n_heads = 8
LOG_EVERY = 5
max_decoding_time_step = 40
# base_path = "./drive/My Drive/Machine Translation/"
base_path = "./"
src_vocab_path = base_path+"NMTtokenizers/spacetoken_vocab_files/vocab_newa.json"
tgt_vocab_path = base_path+"NMTtokenizers/spacetoken_vocab_files/vocab_eng.json"
if not (os.path.exists(src_vocab_path) or os.path.exists(tgt_vocab_path)):
    newa_file = base_path+"dataset/src_train.txt"
    eng_file = base_path+"dataset/tgt_train.txt"
    SpaceTokenizer.create_vocab(eng_file, tgt_vocab_path)
    SpaceTokenizer.create_vocab(newa_file, src_vocab_path)
    print("Vocabulary created....")

def collate_fn(batch):
    src_list, tgt_list = [], []
    for _src, _tgt in batch:
        src_list.append(_src)
        tgt_list.append(_tgt)
    return src_list, tgt_list


@click.group(context_settings = CONTEXT_SETTINGS)
@click.version_option(version='1.0.0')
def task():
    ''' This is the documentation of the main file. This is the reference for executing this file.'''
    pass

@task.command()
@click.option('--src_train', default=base_path+"dataset/src_train.txt", help="train source file path")
@click.option('--tgt_train', default=base_path+"dataset/tgt_train.txt", help="train target file path")
@click.option('--src_valid', default=base_path+"dataset/src_valid.txt", help="validation source file path")
@click.option('--tgt_valid', default=base_path+"dataset/tgt_valid.txt", help="validation target file path")
@click.option('--best_model', default=base_path+"best_model/model.pt", help="best model file path")
@click.option('--model', default="lstm", help="transformer or lstm")
@click.option('--tokenizer', default="space_tokenizer", help="space_tokenizer or bert_tokenizer")
@click.option('--checkpoint_path', default=base_path+"checkpoint/model_ckpt.pt", help=" model check point files path")
@click.option('--seed', default=123, help="manual seed value (default=123)")
def train(**kwargs):
    print("loading dataset")
    train_dataset = NMTDataset(kwargs["src_train"], kwargs["tgt_train"])
    valid_dataset = NMTDataset(kwargs["src_valid"], kwargs["tgt_valid"])
    print("Dataset loaded successfully.")

    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_dl = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    tokenizer = SpaceTokenizer(base_path+"NMTtokenizers/spacetoken_vocab_files/vocab_newa.json", 
                base_path+"NMTtokenizers/spacetoken_vocab_files/vocab_eng.json"
                ) if kwargs["tokenizer"] == "space_tokenizer" else BertTokenizer(
                    base_path+"NMTtokenizers/wordpiece_vocab_files/vocab_newa.json", 
                    base_path+"NMTtokenizers/wordpiece_vocab_files/vocab_eng.json"
                )
    if kwargs['model'] == 'transformer':
        model = TransformerModel(len(tokenizer.src_vocab), len(tokenizer.tgt_vocab), embed_size, 
                n_heads, dropout=dropout_rate)
    else:
        model = Seq2Seq(embed_size, hidden_size, tokenizer, dropout_rate=dropout_rate, n_layers=n_layers)
    # criterion = nn.CrossEntropyLoss()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=3e-3)
    model.to(device)
    model  = trainer(model, optimizer, train_dl, valid_dl, BATCH_SIZE, epoch, 
                            device, LOG_EVERY, kwargs["checkpoint_path"], kwargs["best_model"], 
                            beam_size, max_decoding_time_step)

@task.command()
@click.option('--src_test', default=base_path+"dataset/src_test.txt", help="test source file path")
@click.option('--tgt_test', default=base_path+"dataset/tgt_test.txt", help="test target file path")
@click.option('--best_model', default=base_path+"best_model/model.pt", help="best model file path")
@click.option('--tokenizer', default="space_tokenizer", help="space_tokenizer or bert_tokenizer")
def test(**kwargs):
    print("loading dataset")
    test_dataset = NMTDataset(kwargs["src_test"], kwargs["tgt_test"])
    print("Dataset loaded successfully.")
    test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    tokenizer = SpaceTokenizer(base_path+"NMTtokenizers/spacetoken_vocab_files/vocab_newa.json", 
                base_path+"NMTtokenizers/spacetoken_vocab_files/vocab_eng.json"
                ) if kwargs["tokenizer"] == "space_tokenizer" else BertTokenizer(
                    base_path+"NMTtokenizers/wordpiece_vocab_files/vocab_newa.json", 
                    base_path+"NMTtokenizers/wordpiece_vocab_files/vocab_eng.json"
                )
    model = Seq2Seq(embed_size, hidden_size, tokenizer, dropout_rate=dropout_rate, n_layers=n_layers)
    model.to(device)
    model, _, _, _ = load_checkpt(model, kwargs['best_model'], device)
    eval_start_time = time.time()
    test_loss, bleu_score = evaluate(model, test_dl, 0, device, BATCH_SIZE, beam_size, max_decoding_time_step)
    print(f'Avg. test loss: {test_loss:.5f} | BLEU Score: {bleu_score} | time elapsed: {time.time() - eval_start_time}')

@task.command()
@click.option('--src_file', default=base_path+"dataset/src_file.txt", help="Source file path")
@click.option('--output_file', default=base_path+"dataset/out.txt", help="Output file path")
@click.option('--best_model', default=base_path+"best_model/model.pt", help="best model file path")
@click.option('--tokenizer', default="space_tokenizer", help="space_tokenizer or bert_tokenizer")
def decode(**kwargs):
    src_sent = open_file(kwargs['src_file'])
    tokenizer = SpaceTokenizer(base_path+"NMTtokenizers/spacetoken_vocab_files/vocab_newa.json", 
                base_path+"NMTtokenizers/spacetoken_vocab_files/vocab_eng.json"
                ) if kwargs["tokenizer"] == "space_tokenizer" else BertTokenizer(
                    base_path+"NMTtokenizers/wordpiece_vocab_files/vocab_newa.json", 
                    base_path+"NMTtokenizers/wordpiece_vocab_files/vocab_eng.json"
                )
    model = Seq2Seq(embed_size, hidden_size, tokenizer, dropout_rate=dropout_rate, n_layers=n_layers)
    model.to(device)
    model, _, _, _ = load_checkpt(model, kwargs['best_model'], device)
    hypotheses = SpaceTokenizer.decode(model, src_sent,
                             beam_size=beam_size,
                             max_decoding_time_step=max_decoding_time_step, device=device)

    with open(kwargs['output_file'], 'w', encoding="utf8") as f:
        for src_sent, hyps in zip(src_sent, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')

if __name__ == "__main__":
    task()

