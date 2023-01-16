"""Main script to run experiments."""
import pathlib

from src.config.config import init_config
from src.data.dataloader import get_dataloader
from src.modules.model import MLPMixer
from src.trainer.trainer import Trainer
from src.utils.tools import set_random_seed
from src.utils.tools import load_checkpoint
from src.utils.tools import count_model_parameters


def train_mixer():

    # Get configuration file.
    file_name = "config.yml"
    file_dir = pathlib.Path(__file__).parent.resolve()
    file_path = file_dir / file_name

    # Initialize configuration.
    config = init_config(file_path=file_path)

    # Seed random number generator.
    set_random_seed(seed=config.random_seed)

    # Get dataloader.
    dataloader = get_dataloader(config=config)

    # Get the model.
    model = MLPMixer(config=config)

    count_model_parameters(model=model)

    # Load pre-trained model.
    if config.load_model.is_activated:
        load_checkpoint(
            model=model,
            ckpt_dir=config.dirs.weights,
            model_name=config.load_model.model_name,
        )

    model.to(config.trainer.device)

    print(config)
    trainer = Trainer(model=model, dataloader=dataloader, config=config)
    trainer.run()

    print("Experiment finished.")


if __name__ == "__main__":
    train_mixer()
