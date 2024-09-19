import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *  # noqa: F403
import utils
from nltk.translate.bleu_score import corpus_bleu
import json
import os
from tqdm.auto import tqdm
import mlflow
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
print(device)

emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5

# Training parameters
start_epoch = 0
epochs = 5 # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 0  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.0  # clip gradients at an absolute value of
alpha_c = 1.0  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.0  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = 'Models/checkpoint_flickr8k_4_cap_per_img_4_min_word_freq.pth.tar'  # path to checkpoint, None if none


mlflow.set_experiment('Image Captioning')
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.start_run(log_system_metrics=True)
mlflow.log_param('emb_dim', emb_dim)
mlflow.log_param('attention_dim', attention_dim)
mlflow.log_param('decoder_dim', decoder_dim)
mlflow.log_param('dropout', dropout)
mlflow.log_param('epochs', epochs)
mlflow.log_param('batch_size', batch_size)
mlflow.log_param('workers', workers)
mlflow.log_param('encoder_lr', encoder_lr)
mlflow.log_param('decoder_lr', decoder_lr)
mlflow.log_param('grad_clip', grad_clip)
mlflow.log_param('alpha_c', alpha_c)
mlflow.log_param('fine_tune_encoder', fine_tune_encoder)
mlflow.log_param('checkpoint', checkpoint)
mlflow.log_param('device', device)


def train_model(data_folder, data_name):
    """
    Training and validation of model.
    """
    global \
        best_bleu4, \
        epochs_since_improvement, \
        checkpoint, \
        start_epoch, \
        fine_tune_encoder, \
        word_map

    word_map_file = os.path.join(data_folder, "WORDMAP_" + data_name + ".json")
    with open(word_map_file, "r") as j:
        word_map = json.load(j)

    if checkpoint is None:
        decoder = DecoderWithAttention(
            attention_dim=attention_dim,
            embed_dim=emb_dim,
            decoder_dim=decoder_dim,
            vocab_size=len(word_map),
            dropout=dropout,
        )


        decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, decoder.parameters()),
            lr=decoder_lr,
        )


        encoder = Encoder()
        encoder.fine_tune(fine_tune=fine_tune_encoder)
        encoder_optimizer = (
            torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr,
            )
            if fine_tune_encoder
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

        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr,
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
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(
            data_folder=data_folder,
            data_name=data_name,
            split="VAL",
            transform=transforms.Compose([normalize]),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )

    for epoch in tqdm(range(start_epoch, epochs)):
        if epochs_since_improvement == 20:
            break

        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            utils.adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                utils.adjust_learning_rate(encoder, 0.8)

        train(
            train_loader=train_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch,
        )

        recent_bleu4 = validate(
            val_loader=val_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
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
        )


def train(
    train_loader,
    encoder,
    decoder,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
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
        # print(pack_padded_sequence(scores, decode_lengths, batch_first=True).data)
        # scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        # targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            utils.clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                utils.clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        top5 = utils.accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        mlflow.log_metric('train_loss', loss.item(), step=i)
        mlflow.log_metric('train_top5', top5, step=i)
        
        
        if i % print_freq == 0:
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


def validate(val_loader, encoder, decoder, criterion):
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
            allcaps= allcaps.to(device)
            
            if encoder is not None:
                imgs = encoder(imgs)

            scores, cap_sorted, decode_lengths, alphas, sort_ind = decoder(
                imgs, caps, caplens
            )

            targets = cap_sorted[:, 1:]

            scores_copy = scores.clone()
            scores= pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = criterion(scores, targets)

            loss += alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            losses.update(loss.item(), sum(decode_lengths))
            top5 = utils.accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            mlflow.log_metric('val_loss', loss.item(), step=i)
            mlflow.log_metric('val_top5', top5, step=i)
            
            start = time.time()

            if i % print_freq == 0:
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
            
            mlflow.log_metric('val_bleu4', bleu4, step=i)
            
            print(
                "* LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n".format(
                    loss=losses, top5=top5accs, bleu=bleu4
                )
            )

    return bleu4


if __name__=='__main__':
    train_model(data_folder='Data', data_name='flickr8k_4_cap_per_img_4_min_word_freq')