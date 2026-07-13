from utils.dataset import create_dataloader
from models.vae import VAE
from trainer.vae_trainer import VAETrainer

import config

loader = create_dataloader(config.TRAIN_DATA)

model = VAE()

trainer = VAETrainer(model, loader)

history = trainer.train(epochs=1)

trainer.save_model("vae_model.pth")

print(history)
