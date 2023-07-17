import torch
from torch.nn import functional as F
from torch import nn

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, d_model, block_size, dropout=0.1):

        super().__init__()
        assert d_model % n_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.att_proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1).bool())

    def forward(self, x):

        q = x
        k = x
        v = x
        B,T,_ = x.shape 
        dk = self.d_model // self.n_heads

        # linear projections
        q = self.query(q) 
        k = self.key(k) 
        v = self.value(v) 

        # add number of heads
        q = q.view(B,T,self.n_heads,dk).permute(0,2,1,3)   # B,T,h,dk
        k = k.view(B,T,self.n_heads,dk).permute(0,2,1,3)  
        v = v.view(B,T,self.n_heads,dk).permute(0,2,1,3)  
        
        # attention 

        x = q @ k.transpose(-2,-1) # B,h,T,dk @ B,h,dk,T --> B,h,T,T
        x = x * dk ** -0.5 # B,h,T,T
        x = x.masked_fill(self.mask, float('-inf')) # B,h,T,T
        x = self.dropout(F.softmax(x, dim=(-1)))
        x = x @ v  # B,h,T,T @ B,T,h,dv --> B,h,T,dv
        B,h,T,dv = x.shape
        x = x.transpose(2,1).contiguous().view(B,T,h*dv) #B,T,C
        out = self.dropout(self.att_proj(x)) # B,T,C

        return out
    
class FeedForward(nn.Module):

    def __init__(self, d_model, dff, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dff = nn.Linear(d_model, dff)
        self.out = nn.Linear(dff, d_model)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):

        x = F.gelu(self.dff(x)) 
        x = self.dropout(x) 
        x = self.out(x)

        return x
class DecoderBlock(nn.Module):

    def __init__(self, n_heads, d_model, dff, dropout=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.attention = MultiHeadAttention(n_heads=n_heads, 
                                            d_model=d_model, 
                                            dropout=dropout)
        self.ffl = FeedForward(d_model, dff)
        self.lnorm = nn.LayerNorm(d_model)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        att_out = self.attention(x)
        x = self.lnorm(x + self.dropout(att_out))
        ffl_out = self.ffl(x)
        x = self.lnorm2(x + self.dropout(ffl_out))
        return x

class Decoder(nn.Module):

    def __init__(self, n_heads, d_model, dff, n_layers, dropout=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.decoder = nn.ModuleList([DecoderBlock(n_heads=n_heads,
                                                 d_model=d_model,
                                                 dff=dff,
                                                 dropout=dropout)
                                                 for l in range(n_layers)])

    def forward(self, x):

        for block in self.decoder:

            x = block(x)

        return x
    
class Model(nn.Module):

    def __init__(self, d_model, vocab_size, block_size, n_layers, n_heads, dff, dropout = 0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.d_model = d_model
        self.block_sizse = block_size

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)
        self.decoder = Decoder(n_heads=n_heads,
                                    d_model=d_model,
                                    dff=dff,
                                    n_layers=n_layers,
                                    dropout=dropout)
        self.out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, targets=None):

        B, T = x.shape
        tok_emb = self.embeddings(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        pos_emb = self.pos_embed(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb
        x = self.decoder(x)
        logits = self.out(x)

        if targets == None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(input=logits, target=targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx