"""
Copyright 2022 Balacoon

Test traced SpeechLM model,
comparing embeddings from audio files
with same content but by different speakers
"""

import torch
import soundfile
import numpy as np

import matplotlib.pylab as plt


def main():
    model = torch.jit.load("traced_gpu_half/speechlm.jit").to(torch.device("cuda"))

    # prepare data for inference
    audio_paths = ["slt_arctic_a0001", "rms_arctic_a0001"]
    audio = []
    for path in audio_paths:
        data, sr = soundfile.read(path + ".wav", dtype="int16")
        assert sr == 16000, "model expects 16khz"
        audio.append(data)
    audio_len = [x.shape[0] for x in audio]
    max_audio_len = max(audio_len)
    audio = [np.concatenate([x, np.zeros((max_audio_len - x.shape[0],), dtype=np.int16)]) for x in audio]
    audio = torch.tensor(audio).cuda()
    audio_len = torch.tensor(audio_len).int().cuda()

    # run inference, print pariwise distance
    embs = model(audio, audio_len).detach().cpu().numpy()
    plt.plot(embs[0])
    plt.plot(embs[1])
    plt.show()


if __name__ == "__main__":
    main()
