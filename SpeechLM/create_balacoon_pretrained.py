"""
Copyright 2022 Balacoon

Script to export SpeechLM model
for annotation of audio with pseudo labels
"""

import os
import argparse
import logging
from typing import Tuple

import torch
import torchaudio
import matplotlib.pylab as plt
from SpeechLM import SpeechLMConfig, SpeechLM


def parse_args():
    ap = argparse.ArgumentParser(description="Traces SpeechLM for psuedo annotation of audio")
    ap.add_argument("--clean-ckpt", default="speechlmp_large_checkpoint_clean.pt", help="cleaned checkpoint shorturl.at/fnwH8")
    ap.add_argument("--full-ckpt", default="speechlmp_large_checkpoint_31_400000.pt", help="full checkpoint shorturl.at/luOQT")
    ap.add_argument("--out-dir", default="exported", help="Where to put traced model")
    ap.add_argument("--use-gpu", action="store_true", help="Whether to trace on GPU")
    args = ap.parse_args()
    return args


class DiscreteSpeechEncoder(torch.nn.Module):
    """
    Wrapper around SpeechLM for tracing.
    Takes audio and produces labels (pseudo-phonemes) for each frame
    """
    def __init__(self, checkpoint):
        super().__init__()
        cfg = SpeechLMConfig(checkpoint['cfg']['model'])
        model = SpeechLM(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        self._model = model
        self._to_normalize_audio = checkpoint['cfg']['task']['normalize']

    def forward(self, audio: torch.Tensor, audio_len: torch.Tensor) -> torch.Tensor:
        """
        ivocation of the model

        Parameters
        ----------
        audio: torch.Tensor
            (batch x samples_num) - batch of input audio with 16khz sampling rate
        audio_len: torch.Tensor
            (batch,) - original len of audio sequences in batch before padding.
            is transformed to padding mask

        Returns
        -------
        labels: torch.Tensor
            (batch x frames_num) - pseudo labels for each frame. frames_num = samples_num / 320
        """
        if self._to_normalize_audio:
            audio = torch.nn.functional.layer_norm(audio, (audio.shape[1],))
        # create padding mask out of original length
        # arange = [[0...max_len], [0...max_len], ...]
        arange = torch.arange(0, audio.size(1), device=audio.device).unsqueeze(0).repeat(audio.size(0), 1)
        # real_len = [[len_1, ... len_1], [len_2, ... len_2], ...]
        real_len = audio_len.unsqueeze(1).repeat(1, audio.size(1))
        padding_mask = arange >= real_len
        # batch x frames x 352, where 352 - number of psuedo classes
        _, _, logits = self._model.extract_features(audio, padding_mask=padding_mask)  
        labels = torch.argmax(logits, 2).int()
        return labels


def load_data(path: str, use_gpu: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    helper function that reads audio and composes audio length tensor
    """
    audio, sr = torchaudio.load(path)
    assert sr == 16000
    if use_gpu:
        audio = audio.cuda()
    audio_len = torch.tensor([audio.size(1)], dtype=torch.int, device=audio.device)
    return audio, audio_len


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if args.use_gpu else "cpu")

    logging.info("Add label weights from full checkpoint to cleaned one")
    checkpoint = torch.load(args.clean_ckpt)
    full = torch.load(args.full_ckpt)
    for i in [0, 1]:
        label_emb_list = full["model"]["label_embs_list." + str(i)]
        logging.info("Adding label_emb_list[{}] with shape: ".format(i) + str(label_emb_list.shape))
        checkpoint["model"]["label_embs_list." + str(i)] = label_emb_list

    logging.info("Creating SpeechLM encoder with discrete output")
    model = DiscreteSpeechEncoder(checkpoint).to(device)

    # load input example
    slt, slt_len = load_data("slt_artic_a0001.wav", args.use_gpu)

    # tracing the model
    traced = torch.jit.trace(model, [slt, slt_len])
    out_path = os.path.join(args.out_dir, "speechlm_large.jit")
    traced.save(out_path)

    # running inference with traced model.
    # same utterance but different speakers, ideally model
    # should produce similar output labels
    slt_labels = traced(slt, slt_len).detach().cpu().numpy()[0]
    rms, rms_len = load_data("rms_artic_a0001.wav", args.use_gpu)
    rms_labels = traced(rms, rms_len).detach().cpu().numpy()[0]
    fig, axs = plt.subplots(2)
    axs[0].plot(rms_labels)
    axs[1].plot(slt_labels)
    plt.show()


if __name__ == "__main__":
    main()

