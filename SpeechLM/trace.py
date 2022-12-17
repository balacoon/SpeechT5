"""
Copyright 2022 Balacoon

Script to export SpeechLM model
for annotation of audio with pseudo labels
"""

import os
import argparse
import logging

import torch
import soundfile
import matplotlib.pylab as plt
from SpeechLM import SpeechLMConfig, SpeechLM


def parse_args():
    ap = argparse.ArgumentParser(description="Traces SpeechLM for psuedo annotation of audio")
    ap.add_argument("--clean-ckpt", default="speechlmp_large_checkpoint_clean.pt", help="cleaned checkpoint shorturl.at/fnwH8")
    ap.add_argument("--full-ckpt", default="speechlmp_large_checkpoint_31_400000.pt", help="full checkpoint shorturl.at/luOQT")
    ap.add_argument("--out-dir", default="./exported", help="Where to put traced model")
    ap.add_argument("--cpu", action="store_true", help="If specified, model is traced on CPU, otherwise GPU")
    ap.add_argument("--full-precision", action="store_true", help="If specified, model is traced in full precision")
    args = ap.parse_args()
    return args


class DiscreteSpeechEncoder(torch.nn.Module):
    """
    Wrapper around SpeechLM for tracing.
    Takes audio and produces labels (pseudo-phonemes) for each frame
    """
    def __init__(self, checkpoint, full_precision: bool = False):
        super().__init__()
        cfg = SpeechLMConfig(checkpoint['cfg']['model'])
        model = SpeechLM(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        self._model = model
        self._to_normalize_audio = checkpoint['cfg']['task']['normalize']
        self._full_precision = full_precision

    def forward(self, audio: torch.Tensor, audio_len: torch.Tensor) -> torch.Tensor:
        """
        ivocation of the model

        Parameters
        ----------
        audio: torch.Tensor
            (batch x samples_num) - batch of input audio with 16khz sampling rate in int16 format
        audio_len: torch.Tensor
            (batch,) - original len of audio sequences in batch before padding.
            is transformed to padding mask

        Returns
        -------
        labels: torch.Tensor
            (batch x frames_num) - pseudo labels for each frame. frames_num = samples_num / 320
        """
        float_type = torch.float32 if self._full_precision else torch.float16
        audio = audio.type(float_type) / 32768.0  # to range (-1, 1)
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


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu" if args.cpu else "cuda")

    logging.info("Add label weights from full checkpoint to cleaned one")
    checkpoint = torch.load(args.clean_ckpt)
    full = torch.load(args.full_ckpt)
    for i in [0, 1]:
        label_emb_list = full["model"]["label_embs_list." + str(i)]
        logging.info("Adding label_emb_list[{}] with shape: ".format(i) + str(label_emb_list.shape))
        checkpoint["model"]["label_embs_list." + str(i)] = label_emb_list

    logging.info("Creating SpeechLM encoder with discrete output")
    model = DiscreteSpeechEncoder(checkpoint).to(device)
    model = model if args.full_precision else model.half()
    model.eval()

    # load input example
    example_wav = "slt_arctic_a0001.wav"
    logging.info("Extracting SpeechLM embeddings from a real audio")
    audio, sample_rate = soundfile.read(example_wav, dtype="int16")
    assert sample_rate == 16000, "SpeechLM works with 16khz audio"
    audio = torch.tensor(audio, device=device, dtype=torch.int16).unsqueeze(0)
    audio = audio if args.cpu else audio.cuda()
    audio_len = torch.tensor([audio.size(1)]).int()
    audio_len = audio_len if args.cpu else audio_len.cuda()

    # tracing the model
    with torch.no_grad():
        # running inference with the model to check output
        emb = model(audio, audio_len).detach().cpu().numpy()
        plt.plot(emb[0])
        plt.show()

        # tracing and storing
        traced = torch.jit.trace(model, [audio, audio_len])
        out_path = os.path.join(args.out_dir, "speechlm.jit")
        traced.save(out_path)
        logging.info("Saved traced SpeechLM to {}".format(out_path))


if __name__ == "__main__":
    main()

