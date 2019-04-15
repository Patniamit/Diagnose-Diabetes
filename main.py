from sklearn.preprocessing import StandardScaler
import pickle
from train.diabetes_trainer import train_and_store_model, predict_diabetes
from utils.argument_parser import ArgumentParser


def main():

    arg_parser = ArgumentParser()
    args = arg_parser.parse()

    if args.run_server:
        predict_diabetes(dataset_path=args.dataset_path, model_path=args.model_path)
    else:
        train_and_store_model(dataset_path=args.dataset_path, model_path=args.model_path)


if __name__ == '__main__':
    main()
