import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,x_dim,n_channels,h_dim,hidden_factor=32):
        super().__init__()
        n_layers = torch.ceil(torch.log2(torch.tensor(x_dim))).type(torch.int)
        self.encode = nn.ModuleList([nn.Conv2d(in_channels=n_channels,
                                               out_channels=hidden_factor,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1),
                                     nn.BatchNorm2d(hidden_factor),
                                     nn.LeakyReLU()
                                     ])
        for i in range(1,n_layers):
            self.encode.extend([nn.Conv2d(in_channels=hidden_factor*i,
                                         out_channels=hidden_factor*(i+1),
                                         kernel_size=3,
                                         stride=2,
                                         padding=1),
                                nn.BatchNorm2d(hidden_factor*(i+1)),
                                nn.LeakyReLU()
                                ])
        self.flat = nn.Flatten()
        self.linear = nn.Linear((n_layers)*hidden_factor,h_dim)

    def forward(self,x):
        for layer in self.encode:
            x = layer(x)
        x = self.linear(self.flat(x))
        return x
    
class EncoderDouble(nn.Module):
    """
    Each time we double the C dimension
    """
    def __init__(self,x_dim,n_channels,h_dim,starting_factor=32):
        super().__init__()
        n_layers = torch.ceil(torch.log2(torch.tensor(x_dim))).type(torch.int)
        self.encode = nn.ModuleList([nn.Conv2d(in_channels=n_channels,
                                               out_channels=starting_factor,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1),
                                     nn.BatchNorm2d(starting_factor),
                                     nn.LeakyReLU()
                                     ])
        for _ in range(1,n_layers):
            self.encode.extend([nn.Conv2d(in_channels=starting_factor,
                                         out_channels=starting_factor*2,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1),
                                nn.BatchNorm2d(starting_factor*2),
                                nn.LeakyReLU()
                                ])
            starting_factor *= 2
        self.flat = nn.Flatten()
        self.linear = nn.Linear(starting_factor,h_dim)

    def forward(self,x):
        for layer in self.encode:
            x = layer(x)
        x = self.linear(self.flat(x))
        return x    
    
class DecoderTwo(nn.Module):
    """
    Takes as input a latent vector of shape (B,H,1,1) and return a
    (B,C,H,W) image where H=W are powers of 2.
    """
    def __init__(self,x_dim,n_channels,h_dim):
        super().__init__()
        n_layers = torch.ceil(torch.log2(torch.tensor(x_dim))).type(torch.int) 
        self.decode = nn.ModuleList()
        for _ in range(n_layers):
            self.decode.extend([nn.ConvTranspose2d(in_channels=h_dim,
                                         out_channels=h_dim // 2,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1),
                                nn.BatchNorm2d(h_dim // 2),
                                nn.LeakyReLU()
                                ])
            h_dim = h_dim // 2
        self.decode.append(nn.Conv2d(h_dim,n_channels,kernel_size= 3, padding= 1))
        self.sig = nn.Sigmoid()

    def forward(self,x):
        for layer in self.decode:
            x = layer(x)
        x = self.sig(x)
        return x

class DecoderMNIST(nn.Module):
    """
    Takes as input a latent vector of shape (B,H,1,1) and return a
    (B,C,28,28) image.
    """    
    def __init__(self,h_dim=512,n_channels=1):
        super().__init__()
        self.decode = nn.Sequential(
                         nn.ConvTranspose2d(h_dim,h_dim//2,kernel_size=3,stride = 2,padding=1,output_padding=1),
                         nn.BatchNorm2d(h_dim//2),
                         nn.LeakyReLU(),
        
                         nn.ConvTranspose2d(h_dim//2,h_dim//4,kernel_size=3,stride = 2,padding=1,output_padding=1),
                         nn.BatchNorm2d(h_dim//4),
                         nn.LeakyReLU(),
            
                         nn.ConvTranspose2d(h_dim//4,h_dim//8,kernel_size=3,stride = 2,padding=1),
                         nn.BatchNorm2d(h_dim//8),
                         nn.LeakyReLU(),
            
                         nn.ConvTranspose2d(h_dim//8,h_dim//16,kernel_size=3,stride = 2,padding=1,output_padding=1),
                         nn.BatchNorm2d(h_dim//16),
                         nn.LeakyReLU(),
            
                         nn.ConvTranspose2d(h_dim//16,h_dim//16,kernel_size=3,stride = 2,padding=1,output_padding=1),
                         nn.BatchNorm2d(h_dim//16),
                         nn.LeakyReLU(),
            
                         nn.Conv2d(h_dim//16,n_channels,kernel_size= 3, padding= 1),
                         nn.Sigmoid()           
        )

    def forward(self,x):
        return self.decode(x)    

def logsumexp(x,axis):
    c = torch.max(x)
    return torch.logsumexp(x - c, axis) + c    