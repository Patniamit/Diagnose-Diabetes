import argparse


class ArgumentParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--run_server",
                                 help="run flask server",
                                 action="store_true")

        self.parser.add_argument("--model_path",
                                 type=str,
                                 default="./pretrained_models/diabetes.pkl",
                                 help="path from where model is to be loaded or to be stored")

        self.parser.add_argument("--dataset_path",
                                 type=str,
                                 default="./data/diab.csv",
                                 help="path to dataset")

    def parse(self):
        return self.parser.parse_args()
