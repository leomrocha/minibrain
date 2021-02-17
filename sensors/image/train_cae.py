
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms, utils
from torchvision import datasets
from torchvision.utils import save_image
from cae import *
from helpers import *


def get_loaders(batch_size, transformation, dataset = datasets.CIFAR100, cuda=True):

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset('../data', train=True, download=True,
                transform=transformation),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset('../data', train=False, transform=transformation),
        batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


# Hyper Parameters
# num_epochs = 5
# batch_size = 100
# learning_rate = 0.001


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 12, 12)
    return x


def main():

    num_epochs = 100
    batch_size = 128
    learning_rate = 0.0001
    #learning_rate = 0.001

    model = CAE(12, 12, 3, 2, 128).cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    transformation = full_resolution_crop
    train_loader, test_loader = get_loaders(batch_size, transformation)


    for epoch in range(num_epochs):
        for i, (img, labels) in enumerate(train_loader):
            img = Variable(img).cuda()
            # ===================forward=====================
            #         print("encoding batch of  images")
            output = model(img)
            #         print("computing loss")
            loss = criterion(output, img)
            # ===================backward====================
            #         print("Backward ")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data[0]))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            in_pic = to_img(img.cpu().data)
            save_image(pic, './cae_results/2x2-out_image_{}.png'.format(epoch))
            save_image(in_pic, './cae_results/2x2-in_image_{}.png'.format(epoch))
        if loss.data[0] < 0.15:  # arbitrary number because I saw that it works well enough
            break
    model.save_model("2x2-layer", "CAE")


if __name__ == "__main__":
    main()
