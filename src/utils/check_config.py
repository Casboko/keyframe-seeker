import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
