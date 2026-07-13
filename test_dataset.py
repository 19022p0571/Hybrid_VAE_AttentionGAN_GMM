from utils.dataset import create_dataloader
import config

loader = create_dataloader(config.TRAIN_DATA)

for batch in loader:
    print(batch.shape)
    break
