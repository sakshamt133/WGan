from torch.utils.data import DataLoader
from dataset import Potato
import utils

d = Potato(utils.path, transform=utils.transform)

train_data = DataLoader(
    d, utils.batch_size
)
