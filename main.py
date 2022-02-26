import torch
from generator import Generator
from discriminator import Discriminator
from training_data import train_data
import utils
from torchvision.utils import save_image
from gradient__penalty import gradient_penalty

gen = Generator(utils.noise_dim)
dis = Discriminator(utils.in_channels)
gen_opt = torch.optim.Adam(gen.parameters(), utils.lr)
dis_opt = torch.optim.Adam(dis.parameters(), utils.lr)

for epoch in range(utils.epochs):
    for real in train_data:
        # for _ in range(utils.dis_epochs):
        noise = torch.randn((real.shape[0], utils.noise_dim, 1, 1))
        fake = gen(noise)
        dis_real_out = dis(real)
        dis_fake_out = dis(fake.detach())

        dis_real_loss = torch.mean(dis_real_out)
        dis_fake_loss = torch.mean(dis_real_out)

        disc_loss = - (dis_real_loss - dis_fake_loss) + utils.alpha * gradient_penalty(real, fake, dis)
        print(f" for epoch {epoch} disc loss is {disc_loss}")
        disc_loss.backward(retain_graph=True)
        dis_opt.step()
        dis_opt.zero_grad()

        disc_gen = dis(fake)
        gen_loss = -(disc_gen.mean())
        if epoch % 2 == 0:
            save_image(fake, f'{epoch}.png')

        print(f"for epoch {epoch} gen loss is {gen_loss}")
        gen_loss.backward(retain_graph=True)
        gen_opt.step()
        gen_opt.zero_grad()
