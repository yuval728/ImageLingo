import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import CaptionDataset
import utils
from nltk.translate.bleu_score import corpus_bleu
import json
import os
from tqdm.auto import tqdm
import mlflow
import warnings
import argparse

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
print(device)


def train_model(
    data_folder, data_name, checkpoint=None, save_dir="checkpoints", args=None
):
    """
    Training and validation of model.
    """
    start_epoch = 0
    epochs_since_improvement = 0
    best_bleu4 = 0

    word_map_file = os.path.join(data_folder, "WORDMAP_" + data_name + ".json")
    with open(word_map_file, "r") as j:
        word_map = json.load(j)

    if checkpoint is None:
        
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        decoder = DecoderWithAttention(
            attention_dim=args.attention_dim,
            embed_dim=args.emb_dim,
            decoder_dim=args.decoder_dim,
            vocab_size=len(word_map),
            dropout=args.dropout,
        )

        decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, decoder.parameters()),
            lr=args.decoder_lr,
        )

        encoder = Encoder(fine_tune=args.fine_tune_encoder)
        encoder_optimizer = (
            torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=args.encoder_lr,
            )
            if args.fine_tune_encoder
            else None
        )

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        epochs_since_improvement = checkpoint["epochs_since_improvement"]
        best_bleu4 = checkpoint["bleu-4"]
        decoder = checkpoint["decoder"]
        decoder_optimizer = checkpoint["decoder_optimizer"]
        encoder = checkpoint["encoder"]
        encoder_optimizer = checkpoint["encoder_optimizer"]

        if args.fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=args.encoder_lr,
            )

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(
            data_folder=data_folder,
            data_name=data_name,
            split="TRAIN",
            transform=transforms.Compose([normalize]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(
            data_folder=data_folder,
            data_name=data_name,
            split="VAL",
            transform=transforms.Compose([normalize]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    for epoch in tqdm(range(start_epoch, args.epochs)):
        if epochs_since_improvement == 20:
            break

        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            utils.adjust_learning_rate(decoder_optimizer, 0.8)
            if args.fine_tune_encoder:
                utils.adjust_learning_rate(encoder_optimizer, 0.8)

        train(
            train_loader=train_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch,
            args=args,
        )

        recent_bleu4 = validate(
            val_loader=val_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            args=args,
            word_map=word_map,
        )

        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        else:
            epochs_since_improvement = 0

        utils.save_checkpoint(
            data_name,
            epoch,
            epochs_since_improvement,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            recent_bleu4,
            is_best,
            model_dir=save_dir,
        )


def train(
    train_loader,
    encoder,
    decoder,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
    args,
):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = utils.AverageMeter()  # forward prop. + back prop. time
    data_time = utils.AverageMeter()  # data loading time
    losses = utils.AverageMeter()  # loss (per word decoded)
    top5accs = utils.AverageMeter()  # top5 utils.accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
            imgs, caps, caplens
        )

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss = loss + args.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

        # torch.autograd.set_detect_anomaly(True)

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()

        loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            utils.clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_optimizer is not None:
                utils.clip_gradient(encoder_optimizer, args.grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        top5 = utils.accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        mlflow.log_metric("train_loss", loss.item(), step=i)
        mlflow.log_metric("train_top5", top5, step=i)

        if i % args.print_freq == 0:
            print(
                "\nEpoch: [{0}][{1}/{2}]\t"
                "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top5=top5accs,
                )
            )


def validate(val_loader, encoder, decoder, criterion, args, word_map):
    """
    Performs one epoch's validation

    :param val_loader: Dataset for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer

    :return: BLEU-4 score
    """

    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top5accs = utils.AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()

    with torch.inference_mode():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            allcaps = allcaps.to(device)

            if encoder is not None:
                imgs = encoder(imgs)

            scores, cap_sorted, decode_lengths, alphas, sort_ind = decoder(
                imgs, caps, caplens
            )

            targets = cap_sorted[:, 1:]

            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True
            ).data

            loss = criterion(scores, targets)

            loss += args.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            losses.update(loss.item(), sum(decode_lengths))
            top5 = utils.accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            mlflow.log_metric("val_loss", loss.item(), step=i)
            mlflow.log_metric("val_top5", top5, step=i)

            start = time.time()

            if i % args.print_freq == 0:
                print(
                    "\nValidation: [{0}/{1}]\t"
                    "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top5=top5accs,
                    )
                )

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # references
            allcaps = allcaps[sort_ind]

            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(
                        lambda c: [
                            w
                            for w in c
                            if w not in {word_map["start"], word_map["<pad>"]}
                        ],
                        img_caps,
                    )
                )
                references.append(img_captions)

            # Hypotheses

            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][: decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        bleu4 = corpus_bleu(references, hypotheses)

        mlflow.log_metric("val_bleu4", bleu4, step=i)

        print(
            "* LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n".format(
                loss=losses, top5=top5accs, bleu=bleu4
            )
        )

    return bleu4


def parse_args():
    parser = argparse.ArgumentParser(description="Train Image Captioning Model")
    parser.add_argument(
        "--data_folder",
        default="data",
        help="folder with data files saved by create_input_files.py",
    )
    parser.add_argument(
        "--data_name",
        default="flickr8k_4_cap_per_img_4_min_word_freq",
        help="base name shared by data files",
    )
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint")
    parser.add_argument(
        "--save_dir", default="checkpoints", help="directory to save checkpoints"
    )

    # Hyperparameters
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--emb_dim", type=int, default=512, help="dimension of word embeddings"
    )
    parser.add_argument(
        "--attention_dim",
        type=int,
        default=512,
        help="dimension of attention linear layers",
    )
    parser.add_argument(
        "--decoder_dim", type=int, default=512, help="dimension of decoder RNN"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=160, help="batch size")
    parser.add_argument(
        "--workers", type=int, default=0, help="number of workers for data loading"
    )
    parser.add_argument(
        "--encoder_lr", type=float, default=1e-4, help="learning rate for encoder"
    )
    parser.add_argument(
        "--decoder_lr", type=float, default=4e-4, help="learning rate for decoder"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=5.0, help="gradient clipping value"
    )
    parser.add_argument(
        "--alpha_c",
        type=float,
        default=1.0,
        help="regularization parameter for attention",
    )
    parser.add_argument(
        "--fine_tune_encoder",
        action="store_true",
        help="whether to fine-tune the encoder",
        
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="how often to print stats during training",
    )

    # MLflow-related arguments
    parser.add_argument(
        "--mlflow_experiment", default="ImageLingo", help="MLflow experiment name"
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument("--run_id", default=None, help="MLflow run ID (optional)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # MLflow setup
    mlflow.set_experiment(args.mlflow_experiment)
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    if args.run_id:
        mlflow.start_run(run_id=args.run_id)
    else:
        mlflow.start_run()

    # Log parameters
    mlflow.log_params(vars(args))

    print(args.fine_tune_encoder)
    train_model(
        data_folder=args.data_folder,
        data_name=args.data_name,
        checkpoint=args.checkpoint,
        save_dir=args.save_dir,
        args=args,
    )
