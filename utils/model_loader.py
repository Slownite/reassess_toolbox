#!/usr/bin/env python3

import pathlib
import requests
import torch


class ModelLoader:
    def __init__(self, url, local_filename: pathlib.Path):
        self.url = url
        self.local_filename = local_filename

    def download_weights(self):
        if not self.local_filename.exists():
            print("Downloading weights...")
            response = requests.get(self.url)
            with open(self.local_filename, "wb") as f:
                f.write(response.content)
            print("Download complete.")
        else:
            print("Weights already downloaded, using cached file.")

    def load_model(self, model):
        self.download_weights()
        model.load_state_dict(torch.load(self.local_filename))
        print("Model weights loaded successfully.")
        return model
