import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import os



device = "cpu" 
torch.manual_seed(1337)   # leave this as is (seed does NOT affect performance)

batch_size = 16          
block_size = 64         
eval_interval = 1000
eval_iters = 100         
epoch = 100000        

n_embed = 256            
n_head = 4               
n_layer = 4              
dropout = 0.2



class Head(nn.Module):

	def __init__(self, head_size):

		super().__init__()
		self.key = nn.Linear(n_embed, head_size, bias= False)
		self.query = nn.Linear(n_embed, head_size, bias= False)
		self.value = nn.Linear(n_embed, head_size, bias= False)
		self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

		self.dropout = nn.Dropout(dropout)

	def forward(self, x):

		B,T,C = x.shape

		k =  self.key(x)
		q = self.query(x)

		wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
		wei = wei.masked_fill(self.tril[:T,:T] == 0, float("-inf"))
		wei =  F.softmax(wei,dim = -1)
		wei = self.dropout(wei)
		v =  self.value(x)
		out = wei @ v 

		return out

class MultiHeadAttention(nn.Module):

	def __init__(self, num_heads, head_size):

		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
		self.proj = nn.Linear(n_embed, n_embed)
		self.dropout = nn.Dropout(dropout)
	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim = -1)
		out = self.dropout(self.proj(out))
		return out

class FeedForward(nn.Module):

	def __init__(self, n_embed):

		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embed,4 * n_embed),
			nn.ReLU(),
			nn.Linear(4 * n_embed,n_embed),
			nn.Dropout(dropout),
			)

	def forward(self, x):

		return self.net(x)

class Block(nn.Module):

	def __init__(self, n_embed, n_head):

		super().__init__()

		head_size = n_embed // n_head
		self.sa = MultiHeadAttention(n_head, head_size)
		self.ffwd = FeedForward(n_embed)
		self.ln1 = nn.LayerNorm(n_embed)
		self.ln2 = nn.LayerNorm(n_embed)


	def forward(self, x):
		x = x + self.sa(self.ln1(x))
		x = x + self.ffwd(self.ln2(x))
		return x


class BigramLanguageModel(nn.Module):

	def __init__(self, vocab_size):

		super().__init__()
		self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
		self.position_embedding_table = nn.Embedding(block_size, n_embed)
		self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embed)
		self.lm_head = nn.Linear(n_embed, vocab_size)

	def forward(self, idx, targets = None):

		B, T = idx.shape

		tok_emb =  self.token_embedding_table(idx)
		pos_emb = self.position_embedding_table(torch.arange(T,device = device))
		x = tok_emb + pos_emb
		x = self.blocks(x)
		x = self.ln_f(x)
		logits = self.lm_head(x)

		if targets == None:
			loss = None

		else:

			B, T, C = logits.shape
			logits = logits.view(B*T, C)
			targets = targets.view(B*T)

			loss = F.cross_entropy(logits, targets)

		return logits, loss

	def generate(self, idx, max_new_tokens):

		for _ in range(max_new_tokens):

			idx_cond = idx[:, -block_size:]

			logits, loss = self(idx_cond)

			logits = logits[:, -1 , :]

			probs = F.softmax(logits, dim = -1)

			idx_next = torch.multinomial(probs, num_samples=1)

			idx = torch.cat((idx, idx_next), dim = 1)

		return idx

def File_Accessing(Path):

	with open(Path, "r", encoding="utf-8") as Data_file:

		Data_Set = Data_file.read()
		Chars = sorted(list(set(Data_Set)))

	return Chars, Data_Set

def Mapping_Chars(Char):

	global ItoS, StoI

	StoI = {}   # String to Int

	for Index, Character in enumerate(Char):
		StoI[Character] = Index

	ItoS = {}  #Int to String

	for Index, Character in enumerate(Char):
		ItoS[Index] = Character

def Encoder(String):

	Encoded_Lis = []

	for Character in String:

		Encoded_Lis.append(StoI[Character])

	return Encoded_Lis

def Decoder(Index_Lis):

	Decoded_Lis = []

	for index in Index_Lis:

		Decoded_Lis.append(ItoS[index])

	return "".join(Decoded_Lis)

def Split_Data(Data):

	global  train_data, eval_data

	n = int(0.9*len(Data))

	train_data = Data[:n]
	eval_data = Data[n:]
	
def Batch_Block( Split):

	if Split == "train":

		data = train_data

	else:
		data = eval_data


	ix = torch.randint(len(data) - block_size, (batch_size, ))
	x = torch.stack([data[i:i+block_size] for i in ix])
	y = torch.stack([data[i+1:i+block_size+1] for i in ix])

	return x , y

def train(Split, model, optimizer, start_step=0):

    start_time = time.time()
    checkpoint_path = "checkpoint.pt"

    for steps in range(start_step, epoch):

        elapsed = time.time() - start_time
        progress = (steps + 1) / epoch
        estimated_total = elapsed / progress
        remaining = estimated_total - elapsed

        print(f"\rProgress: {progress*100:.2f}% | "
              f"Elapsed: {elapsed/60:.2f} min | "
              f"Remaining: {remaining/60:.2f} min", end="")

        if steps % eval_interval == 0:
            losses = estimate_loss()
            print(f"\nstep {steps}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['eval']:.4f}")

            # ---- SAVE CHECKPOINT ----
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": steps,
                "rng_state": torch.get_rng_state()
            }, checkpoint_path)

        xb, yb = Batch_Block(Split)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\nTraining complete.")



@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for Split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = Batch_Block(Split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[Split] = losses.mean()
    model.train()
    return out

def main():

    global model

    Path = "input.txt"
    Chars, Data_Set = File_Accessing(Path)
    vocab_size = len(Chars)
    print(vocab_size)

    Mapping_Chars(Chars)

    Data_Token = torch.tensor(Encoder(Data_Set), dtype=torch.long)

    Split_Data(Data_Token)
    Split = "train"

    model = BigramLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    checkpoint_path = "checkpoint.pt"
    start_step = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"]
        torch.set_rng_state(checkpoint["rng_state"])
        print("Checkpoint loaded. Resuming training...")

    train(Split, model, optimizer, start_step)

    print(Decoder(
        model.generate(
            idx=torch.zeros((1, 1), dtype=torch.long),
            max_new_tokens=1500
        )[0].tolist()
    ))


main()