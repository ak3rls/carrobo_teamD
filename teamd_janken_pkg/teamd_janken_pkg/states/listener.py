# import rclpy
# from rclpy.node import Node
import sounddevice as sd
from yasmin import Blackboard
from yasmin import State
import numpy as np
import torch

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)


class Whisper_state(State):
    def __init__(self):
        super().__init__(["success"])
        sd.default.device = 9
        model_name = "openai/whisper-small"
        device = 0
        dtype = torch.float32

        print("モデルロード中")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            dtype = dtype
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name
        )

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=device,
            dtype=dtype
        )
        print("モデルロード完了")


    def execute(self, sampling_rate = 16000):
        while True:
            try:
                input("enterキーでスタート")

                hantei = ("最初はぐー","最初はグー") #判定する言語の設定
                prompt_text = "最初はグー"

                print("enterKeyでストップ")

                frames = []


                #めざせ1.5秒

                with sd.InputStream(
                    samplerate = sampling_rate,
                    channels = 1,
                    dtype = "int16",
                    callback = lambda indata, *_: frames.append(indata.copy())
                ):
                    input()

                # whisperに渡すデータとして変換
                speech = np.concatenate(frames, axis=0 ).flatten().astype(np.float32) / 32768.0

                prompt_ids = torch.tensor(self.processor.get_prompt_ids(prompt_text)).to(self.model.device)

                result = self.pipe(
                    {"array": speech, "sampling_rate": sampling_rate},
                    generate_kwargs = {
                        "language": "japanese",
                        "prompt_ids": prompt_ids,
                    },
                )

                text = result["text"]

        
                print("ninnsiki",text)

                if any(word in text for word in hantei):
                    print("次のステートへ")
                    return "success"
                    break
                else: 
                    print("もう一度")

            except KeyboardInterrupt:
                print("finish")
                break

if __name__ == "__main__":
    whisper = Whisper_state()
    whisper.execute()

    
