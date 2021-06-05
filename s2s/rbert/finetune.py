import logging
import fastwer
import numpy as np
import wandb
import torch.multiprocessing
from transformers import RobertaConfig, EncoderDecoderConfig
torch.multiprocessing.set_sharing_strategy('file_system')

import pandas as pd
from simpletransformers.seq2seq import (
    Seq2SeqModel,
    Seq2SeqArgs,
)

def count_matches(labels, preds):
    return sum(
        [
            1 if label == pred else 0
            for label, pred in zip(labels, preds)
        ]
    )

def get_wer(labels, preds):
    return np.mean(
        [
            fastwer.score_sent(pred, label)
            for label, pred in zip(labels, preds)
        ]
    )

def main():
    model_args = Seq2SeqArgs()
    model_args.num_train_epochs = 20
    # model_args.no_save = True
    model_args.evaluate_generated_text = True
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_verbose = False
    model_args.tensorboard_dir = "runs"
    model_args.max_length = 50
    model_args.train_batch_size=20
    model_args.overwrite_output_dir=True
    model_args.wandb_project = "cs224u"
    model_args.use_multiprocessed_decoding = True

    # model = EncoderDecoderModel(config=config)
    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name="facebook/bart-large-cnn",
        args=model_args,
        use_cuda=True,
    )
#     model = Seq2SeqModel(encoder_type="roberta", encoder_name="./outputs/checkpoint-9702-epoch-6/encoder", decoder_name="./outputs/checkpoint-9702-epoch-6/decoder", args=model_args, config=config, use_cuda=True)
    
    train_df = pd.read_pickle("train.pkl")
    train_df = train_df.dropna()
    dev_df = pd.read_pickle("dev.pkl")

    # Train the model
    wandb.init(project='cs224u', entity='gbanerje')

    # 2. Save model inputs and hyperparameters
    config = wandb.config
    config.learning_rate = 0.01

    # Model training here

    model.train_model(
        train_df, eval_data=dev_df, matches=count_matches, wer=get_wer, show_running_loss=True, args={'fp16': False}
    )

    wandb.join()

    # # Evaluate the model
    results = model.eval_model(dev_df)
    
    
if __name__=="__main__":
    main()