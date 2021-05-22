from numpy.lib.function_base import append
import torch
from models.transformer import *
from dataset import NMTDataset
import click
from torch.utils.data import DataLoader
from NMTtokenizers.tokenizer import *
from utils import *
import time
from models.transformer import TransformerModel
from beam import greedy_decode, beam
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
embed_size = 192
hidden_size = 128
dropout_rate = 0.1  
n_layers = 2
beam_size = 5
epoch = 25
n_heads = 8
LOG_EVERY = 1
max_decoding_time_step = 15

CONTEXT_SETTINGS = dict(help_option_names = ['-h', '--help'])
base_path = "./"
# base_path = "./drive/My Drive/Machine Translation/"

src_vocab_path = base_path+"NMTtokenizers/spacetoken_vocab_files/vocab_newa.json"
tgt_vocab_path = base_path+"NMTtokenizers/spacetoken_vocab_files/vocab_eng.json"

def collate_fn(batch):
    src_list, tgt_list = [], []
    for _src, _tgt in batch:
        src_list.append(_src)
        tgt_list.append(_tgt)
    return src_list, tgt_list

def compute_bleu_score(output, labels):
    refs = SpaceTokenizer.tokenize(labels, batch=True, wrap_inner_list = True)
    output_tokens = SpaceTokenizer.tokenize(output, batch=True)
    weights = (1.0/2.0, 1.0/2.0, )
    score = corpus_bleu(refs, output_tokens, smoothing_function=SmoothingFunction(epsilon=1e-10).method1, weights=weights)
    return score

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
@click.option('--model', default="transformer", help="transformer or lstm")
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

    model = TransformerModel(len(tokenizer.src_vocab), len(tokenizer.tgt_vocab), tokenizer, embed_size, 
                n_heads, dropout=dropout_rate)

    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-3)
    
    train_model(model, optimizer, criterion, train_dl, valid_dl, BATCH_SIZE, epoch, 
                            device, LOG_EVERY, kwargs["checkpoint_path"], kwargs["best_model"], 
                            beam_size, max_decoding_time_step)
                            
def train_model(model, optimizer, criterion, train_dl, valid_dl, batch_size, epoch, device, LOG_EVERY, checkpt_path, best_model_path, beam_size, max_decoding_time_step):
    eval_loss = float('inf')
    start_epoch = 0
    if os.path.exists(checkpt_path):
        model, optimizer, eval_loss, start_epoch = load_checkpt(model, checkpt_path, device, optimizer)
        print(f"Loading model from checkpoint with start epoch: {start_epoch} and loss: {eval_loss}")

    model.to(device)
    best_eval_loss = eval_loss
    print("Model training started...")
    for epoch in range(start_epoch, epoch):
        print(f"Epoch {epoch} running...")
        epoch_start_time = time.time()
        epoch_train_loss = 0
        epoch_eval_loss = 0
        model.train()
        bleu_score = 0
        for step, batch in enumerate(train_dl):
            src_tensor, tgt_tensor, _, _ = model.tokenizer.encode(batch, device, return_tensor=True)
            src_tensor = src_tensor.transpose(0,1)
            tgt_tensor = tgt_tensor.transpose(0,1)
            trg_input = tgt_tensor[:, :-1]
            targets = tgt_tensor[:, 1:].contiguous().view(-1)
            
            optimizer.zero_grad()
            # model.zero_grad()
            preds = model(src_tensor, trg_input.to(device), device)

            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()/batch_size

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(valid_dl):
                src_tensor, tgt_tensor, _, _ = model.tokenizer.encode(batch, device, return_tensor=True)
                src_tensor = src_tensor.transpose(0,1)
                tgt_tensor = tgt_tensor.transpose(0,1)
                trg_input = tgt_tensor[:, :-1]
                targets = tgt_tensor[:, 1:].contiguous().view(-1)
                preds = model(src_tensor, trg_input.to(device), device)
                
                loss = criterion(preds, targets)
                epoch_eval_loss += loss.item()/batch_size

                translator = Translator(model, beam_size, max_decoding_time_step, 
                        model.tokenizer.src_vocab['[PAD]'], model.tokenizer.tgt_vocab['[PAD]'], 
                        model.tokenizer.tgt_vocab['[SOS]'], model.tokenizer.tgt_vocab['[EOS]']).to(device)
                output = []
                for src in src_tensor:
                    pred_seq = translator.translate_sentence(src.view(1, -1), device)
                    pred_line = ' '.join(model.tokenizer.tgt_vocab.id2word[idx] for idx in pred_seq)
                    pred_line = pred_line.replace('[SOS]', '').replace('[EOS]', '')
                    output.append(pred_line)
                    
                # print(output)
                score = compute_bleu_score(output, batch[1])
                bleu_score += score
                # print(f"Bleu Score: {bleu_score}")

        print(f'Epoch: {epoch} Compeleted | avg. train loss: {epoch_train_loss/len(train_dl)} | time elapsed: {time.time() - epoch_start_time}')
        print(f'Epoch: {epoch} Compeleted | avg. eval loss: {epoch_eval_loss/len(valid_dl)} | BLEU Score: {bleu_score/len(valid_dl)} | time elapsed: {time.time() - epoch_start_time}')
        # exit()
        check_pt = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eval_loss': epoch_eval_loss,
            'batch_size': batch_size,
        }
        check_pt_time = time.time()
        print("Saving Checkpoint.......")
        if epoch_eval_loss < best_eval_loss:
            print("New best model found")
            best_eval_loss = epoch_eval_loss
            save_model_checkpt(check_pt, True, checkpt_path, best_model_path)
        else:
            save_model_checkpt(check_pt, False, checkpt_path, best_model_path)  
        print(f"Checkpoint saved successfully with time: {time.time() - check_pt_time}")

# @task.command()
# @click.option('--src_test', default=base_path+"dataset/src_test.txt", help="test source file path")
# @click.option('--tgt_test', default=base_path+"dataset/tgt_test.txt", help="test target file path")
# @click.option('--best_model', default=base_path+"best_model/model.pt", help="best model file path")
# @click.option('--tokenizer', default="space_tokenizer", help="space_tokenizer or bert_tokenizer")
# def test(**kwargs):
#     print("loading dataset")
#     test_dataset = NMTDataset(kwargs["src_test"], kwargs["tgt_test"])
#     print("Dataset loaded successfully.")
#     test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
#     tokenizer = SpaceTokenizer(base_path+"NMTtokenizers/spacetoken_vocab_files/vocab_newa.json", 
#                 base_path+"NMTtokenizers/spacetoken_vocab_files/vocab_eng.json"
#                 ) if kwargs["tokenizer"] == "space_tokenizer" else BertTokenizer(
#                     base_path+"NMTtokenizers/wordpiece_vocab_files/vocab_newa.json", 
#                     base_path+"NMTtokenizers/wordpiece_vocab_files/vocab_eng.json"
#                 )
#     model = Seq2Seq(embed_size, hidden_size, tokenizer, dropout_rate=dropout_rate, n_layers=n_layers)
#     model.to(device)
#     model, _, _, _ = load_checkpt(model, kwargs['best_model'], device)
#     eval_start_time = time.time()
#     test_loss, bleu_score = evaluate(model, test_dl, 0, device, BATCH_SIZE, beam_size, max_decoding_time_step)
#     print(f'Avg. test loss: {test_loss:.5f} | BLEU Score: {bleu_score} | time elapsed: {time.time() - eval_start_time}')

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
    model = TransformerModel(len(tokenizer.src_vocab), len(tokenizer.tgt_vocab), tokenizer, embed_size, 
                n_heads, dropout=dropout_rate)

    model.to(device)
    model, _, _, _ = load_checkpt(model, kwargs['best_model'], device)
    src_tensor, _ = tokenizer.encode(src_sent, device, return_tensor=True)

    # predictor = Predictor(model, max_decoding_time_step, beam_size)

    # translator = Translator(model, beam_size, max_decoding_time_step, 
    #                     model.tokenizer.src_vocab['[PAD]'], model.tokenizer.tgt_vocab['[PAD]'], 
    #                     model.tokenizer.tgt_vocab['[SOS]'], model.tokenizer.tgt_vocab['[EOS]']).to(device)
    output = []
    for src in src_tensor:
        # pred_seq = translator.translate_sentence(src.view(1, -1), device)
        pred_seq = greedy_decode(model, src.view(1, -1), max_decoding_time_step, model.tokenizer.tgt_vocab['[SOS]'], 
                model.tokenizer.src_vocab['[PAD]'], model.tokenizer.tgt_vocab['[EOS]'], device)
        pred_line = ' '.join(model.tokenizer.tgt_vocab.id2word[idx] for idx in pred_seq)
        pred_line = pred_line.replace('[SOS]', '').replace('[EOS]', '')
        output.append(pred_line)
        print(pred_line)
    # print(output)

if __name__ == "__main__":
    task()