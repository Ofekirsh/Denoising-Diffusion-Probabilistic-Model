import argparse
import torch, torchvision
from torchvision import transforms, utils
import numpy as np
from DDPM import DDPM
import matplotlib.pyplot as plt
import pickle as pkl

def train(model, trainloader, optimizer, epoch,device):
    loss_epoch = 0
    model.train()  # set to training mode
    for image, target in trainloader:
        noise = torch.randn_like(image).to(device)
        image = image.to(device)
        target = target.to(device)

        # Choose a random timestep for each sample in the batch
        timesteps = torch.randint(0, model.timesteps, (image.size(0),), device=device).long()

        # Forward pass: Predict noise from the model
        predicted_noise = model(image, noise, timesteps, target)

        # Compute loss (MSE between predicted and true noise)
        loss = model.loss(predicted_noise, noise)
        loss_epoch += loss.item()
        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Training completed. Loss: {loss.item()}")
    return loss_epoch/len(trainloader)

def sample(model,epoch):
    model.eval()
    with torch.no_grad():
        samples = model.sample()*0.5+0.5
        samples.clamp_(0., 1.)
        grid_image = utils.make_grid(samples, nrow=10, padding=2)
        grid_image = grid_image.numpy().transpose((1, 2, 0))  # Convert from CxHxW to HxWxC
        plt.imshow(grid_image, cmap='gray')
        plt.axis('off')  # Hide the axes
        filename = './samples/MNIST_' +str(epoch)+'.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)


def ddim_sample(model, n_steps):
    model.eval()

    std = 0.0

    steps = torch.linspace(0, model.timesteps - 1, steps=n_steps + 1).long().to(model.device)
    steps = steps.tolist()
    steps = list(reversed(steps))

    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10).to(model.device)
    x = torch.randn(100, 1, 28, 28).to(model.device)

    for i in range(len(steps) - 1):
        t_step = steps[i]
        t_prev_step = steps[i + 1]

        t_tensor = torch.full((x.size(0),), t_step, dtype=torch.long).to(model.device)

        noise_pred = model.model(x, t_tensor, y, uncond=False)

        alpha_bar_t = model.alpha_bars[t_step].reshape(-1, 1, 1, 1)
        alpha_bar_t_prev = model.alpha_bars[t_prev_step].reshape(-1, 1, 1, 1)

        predicted_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        predicted_x0 = predicted_x0.clamp(-1, 1)

        dir_to_xt = torch.sqrt(1 - alpha_bar_t_prev - std ** 2) * noise_pred

        noise = torch.randn_like(x).to(model.device) if std > 0 else 0

        x = torch.sqrt(alpha_bar_t_prev) * predicted_x0 + dir_to_xt + std * noise

        x = x.clamp(-1, 1)

    return x.cpu()

def ddim_plot_sample(model, num_steps):
    model.eval()
    with torch.no_grad():
        generated_samples = ddim_sample(model, num_steps)
        normalized_samples = generated_samples * 0.5 + 0.5
        normalized_samples.clamp_(0., 1.)

        grid_image = utils.make_grid(normalized_samples, nrow=10, padding=2)
        grid_image = grid_image.numpy().transpose((1, 2, 0))  # Convert from CxHxW to HxWxC

        plt.imshow(grid_image, cmap='gray')
        plt.axis('off')  # Hide the axes

        output_filename = f'./samples/DDIM_{num_steps}.png'
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])  # [0,1] to [-1,1]

    trainset = torchvision.datasets.MNIST(root='./data/MNIST',
        train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data/MNIST',
        train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
        batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = DDPM(device=device).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr)
    losses = []
    for i in range(args.epochs):
        loss_epoch = train(model,trainloader,optimizer,i,device)
        losses.append(loss_epoch)
        sample(model,i)
    with open("loss_epochs.pkl", "wb") as file:
        pkl.dump(losses, file)

    for k in [5,10,20,50]:
        ddim_plot_sample(model, k)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=30)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)


