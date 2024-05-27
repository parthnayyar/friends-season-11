import os
import sys
import torch
from dataloader import DataLoader

class Head(torch.nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = torch.nn.Linear(n_embd, head_size, bias=False)
        self.query = torch.nn.Linear(n_embd, head_size, bias=False)
        self.value = torch.nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = torch.nn.functional.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(head_size * num_heads, n_embd)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(torch.nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd, n_embd),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x): return self.net(x)

class Block(torch.nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.ln2 = torch.nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class Model(torch.nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, dropout, n_layer): 
        super().__init__()
        self.block_size = block_size
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embd)
        self.blocks = torch.nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = torch.nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = torch.nn.Linear(n_embd, vocab_size)
        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module): 
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding): torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, device, targets=None): 
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None: loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens): 
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            # break if <EPISODE_END> reached
            if idx_next.item() == 1: break
        return idx
    
    def estimate_loss(self, eval_iters, dataloader, batch_size, device, debug=False): 
        with torch.no_grad():
            out = {}
            self.eval()
            for split in ['train', 'val']:
                if debug: print(f'{split} data', end=': ')
                losses = torch.zeros(eval_iters)
                for i in range(eval_iters):
                    X, Y = dataloader.get_batch(split, self.block_size, batch_size, device)
                    _, loss = self(X, device, targets=Y)
                    losses[i] = loss.item()
                    if debug and i%(eval_iters//5)==0: print(f'{i+1}/{eval_iters} done', end=', ') 
                    if debug and i==eval_iters-1: print(f'{i+1}/{eval_iters} done') 
                out[split] = losses.mean()
            self.train()
            return out
    
    def train_model(self, train, val, max_iters, eval_iters, batch_size, lr, device, debug=False):
        if debug: print('training start')
        dataloader = DataLoader(train, val)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        tr_losses, val_losses = [], []
        for i in range(max_iters):
            # every once in a while evaluate the loss on train and val sets
            if i % eval_iters == 0 or iter == max_iters - 1:
                if debug: print(f'Evaluate losses at step: {i}')
                losses = self.estimate_loss(eval_iters, dataloader, batch_size, device, debug=debug)
                tr_loss, val_loss = losses['train'], = losses['val']
                print(f'step {i}: train loss {tr_loss:.4f}, val loss {val_loss:.4f}')
                tr_losses.append(tr_loss)
                val_losses.append(val_loss)
            # sample a batch of data
            xb, yb = dataloader.get_batch('train', self.block_size, batch_size, device)
            # evaluate the loss
            _, loss = self(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if debug: print('training finish')
        return tr_losses, val_losses
