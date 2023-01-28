import torch
import torchvision
from . import train
from .model import MyModel
from .load_data import train_dataloader, mnist_mean, mnist_std, device
import os.path as path
import gradio as gr
import matplotlib.pyplot as plt

my_net = MyModel()
my_net.cuda()
my_net.train()
optimizer = torch.optim.Adam(my_net.parameters(), lr=0.0001 )

model_path = 'trained_net.pt'
# check if net is trained
if path.exists(model_path):
    my_net.load_state_dict(torch.load(model_path))
# else train model and save weights
else:
    train.train_loop(my_net, optimizer, train_dataloader)
    torch.save(my_net.state_dict(), model_path)

# set net to eval mode
my_net.eval()
norm_trans = torchvision.transforms.Normalize(mnist_mean, mnist_std)
print("training/loading complete")


# gradio
def predict(img):
    plt.imshow(img,cmap='gray')
    plt.show()
    x = torch.tensor(img, dtype=torch.float32, device=device)
    # add dummy dimensions
    x = x[None, None, :]
    # normalize between 0 and 1 and then to mnist mean and std
    x /= 255
    x = norm_trans(x)
    with torch.no_grad():
        out = my_net(x)
    results = torch.nn.functional.softmax(out[0], dim=0)
    # return a dictionary with index-digits as key and predictions as values
    predictions = {str(i): results[i].cpu().item() for i in range(10)}
    return predictions


gr.Interface(fn=predict,
             inputs="sketchpad",
             outputs=gr.Label()
             ).launch(inbrowser=True)
