from utils.dataset import create_dataloader
from models.generator import Generator
from models.discriminator import Discriminator
from trainer.gan_trainer import GANTrainer
import config

loader = create_dataloader(config.TRAIN_DATA)

generator = Generator()
discriminator = Discriminator()

trainer = GANTrainer(
    generator,
    discriminator,
    loader
)

print("GAN Trainer created successfully.")
