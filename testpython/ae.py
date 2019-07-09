
class VAE(nn.Module):
    def __init__(self, latent_variable_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2m = nn.Linear(400, latent_variable_dim) # use for mean
        self.fc2s = nn.Linear(400, latent_variable_dim) # use for standard deviation
        
        self.fc3 = nn.Linear(latent_variable_dim, 400)
        self.fc4 = nn.Linear(400, 784)
        
    def reparameterize(self, log_var, mu):
        s = torch.exp(0.5*log_var)
        eps = torch.rand_like(s) # generate a iid standard normal same shape as s
        return eps.mul(s).add_(mu)
        
    def forward(self, input):
        x = input.view(-1, 784)
        x = torch.relu(self.fc1(x))
        log_s = self.fc2s(x)
        m = self.fc2m(x)
        z = self.reparameterize(log_s, m)
        
        x = self.decode(z)
        
        return x, m, log_s
    
    def decode(self, z):
        x = torch.relu(self.fc3(z))
        x = torch.sigmoid(self.fc4(x))
        return x

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)

        self.c3 = nn.Conv2d(32, 32, 3, padding=1)
        self.c4 = nn.Conv2d(32, 32, 3, padding=1)
        

        self.mp = nn.MaxPool2d((2,2), stride= 2)

        self.up = Interpolate(scale_factor= 2, mode='nearest')

        self.cf = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, input):

        #x = input.view(-1, 784)
        x = self.mp(torch.relu(self.c1(input)))

        x = self.mp(torch.relu(self.c2(x)))

        x = self.up(torch.relu(self.c3(x)))

        x = self.up(torch.relu(self.c4(x)))

        x = torch.sigmoid(self.cf(x))
       
        return x









def lossCAE(input_image, recon_image,):
    CE = F.binary_cross_entropy(recon_image, input_image, reduction='sum')
    return CE


def loss(input_image, recon_image, mu, log_var):
    CE = F.binary_cross_entropy(recon_image, input_image.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return KLD + CE

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def show_images(images):
    images = utils.make_grid(images)
    show_image(images[0])

def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()







if __name__ == '__main__':    







#    img_transform = transforms.Compose([
#    transforms.ToTensor()  
#])

#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#   ## training
#    BATCH_SIZE = 100


#    trainset = datasets.MNIST('./data/', train=True, download=True,
#                       transform=img_transform)
    
#    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
#                                              shuffle=True, num_workers=8)
#    # test
#    testset = datasets.MNIST('./data/', train=False, download=False,
#                       transform=img_transform)

#    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
#                                              shuffle=True, num_workers=8) 


    
#    #transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])
#    #trainTransform  = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])
#    #trainset = tv.datasets.CIFAR10(root='./data',  train=True,download=True, transform=transform)
#    #dataloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
#    #testset = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#    #testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

#    #dataiter = iter(trainloader)
#    #images, labels = dataiter.next()
#    #show_images(images)

#    # train
#    noise_factor = 0.5


#    #vae = VAE(40).cuda()

#    #cae = CAE().cuda()

#    #summary(cae, (1,28,28))
    
#    #learning_rate = 1e-4
#    #optimizer = torch.optim.Adam(cae.parameters(), lr=learning_rate)

#    #train_loss = []
#    #for epoch in range(20):
#    #    for i, data in enumerate(trainloader, 0):
            
#    #        images, labels = data
#    #        noise = torch.from_numpy( np.array( noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape), dtype= np.float32))
            

#    #        images_n = images + noise
#    #        images = images.to(device)
#    #        images_n = images_n.to(device)


#    #        optimizer.zero_grad()
#    #        recon_image = cae(images_n)
#    #        l = lossCAE(images, recon_image)
#    #        print('Loss', l)
#    #        l.backward()
#    #        train_loss.append(l.item() / len(images))
#    #        optimizer.step()
#    #plt.plot(train_loss)
#    #plt.show()

#    #torch.save(cae.state_dict(),'d:/pesos_cae.json')


#    cae = CAE().cuda()
#    cae.load_state_dict(torch.load('d:/pesos_cae.json'))
#    cae.eval()

#    with torch.no_grad():
#        for i, data in enumerate(testloader, 0):
#            images, labels = data
#            noise = torch.from_numpy( np.array( noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape), dtype= np.float32))
            

#            images_n = images + noise
#            images = images.to(device)
#            images_n = images_n.to(device)

#            recon_image = cae(images_n)

#            recon_image_ = recon_image.view(BATCH_SIZE, 1, 28, 28)
#            if i % 100 == 0:
#                show_images(recon_image_.cpu())
#                show_images(images.cpu())
#                show_images(images_n.cpu())
