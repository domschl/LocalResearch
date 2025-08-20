import toml
from pathlib import Path
import os

class Settings:
    def __init__(self, config_path: Path = Path(__file__).parent.parent / "web_server_config.toml"):
        config = toml.load(config_path)
        self.server_host: str = config["server"]["host"]
        self.server_port: int = config["server"]["port"]
        self.resources_data_path: Path = (Path(__file__).parent.parent / config["resources"]["data_path"]).resolve()
        # Must be in web-server stuff:
        # self.resources_data_path: Path = os.path.expanduser(config["resources"]["data_path"])
        # self.tls_enabled: bool = config.get("tls", {}).get("enabled", False)
        # self.tls_keyfile: str | None = config.get("tls", {}).get("keyfile")
        # self.tls_certfile: str | None = config.get("tls", {}).get("certfile")

settings = Settings()

if __name__ == "__main__":
    print(f"Server Host: {settings.server_host}")
    print(f"Server Port: {settings.server_port}")
    print(f"Data Path: {settings.resources_data_path}")
