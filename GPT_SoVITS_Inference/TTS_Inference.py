from TTS_infer_pack.text_segmentation_method import get_method
from TTS_infer_pack.TTS import NO_PROMPT_ERROR, TTS, TTS_Config
import torch
import os
import yaml
import soundfile as sf


class TTS_Model_Package:
    def __init__(self, model_package_path: str):
        self.model_package_path = model_package_path
        self.model_version_path = os.path.join(model_package_path, "model_version.yaml")
        self.load_model_config()

    def load_model_config(self):
        with open(self.model_version_path, "r", encoding="utf-8") as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)

    def get_t2s_weights_path(self):
        return os.path.join(self.model_package_path, self.model_config.get("t2s_weights_path", "艾莲-e10.ckpt"))

    def get_vits_weights_path(self):
        return os.path.join(self.model_package_path, self.model_config.get("vits_weights_path", "艾莲_e10_s450_l32.pth"))

    def get_ref_audio_path(self):
        return os.path.join(self.model_package_path, self.model_config.get("ref_audio_path", "嗯，幽灵小姐狼人先生和巨子少女。.wav"))
    
    def get_ref_text_free(self):
        return self.model_config.get("ref_text_free", "嗯，幽灵小姐狼人先生和巨子少女。")
    
    def get_ref_text_free_lang(self):
        return self.model_config.get("ref_text_free_lang", "all_zh")
    
    def get_tts_config(self):
        tts_config = TTS_Config("GPT_SoVITS_Inference/configs/tts_infer.yaml")
        if torch.cuda.is_available():
            device = "cuda"
        # elif torch.backends.mps.is_available():
        #     device = "mps"
        else:
            device = "cpu"
        tts_config.device = device
        tts_config.is_half = self.model_config.get("is_half", True)
        tts_config.update_version(self.model_config.get("version", "v4"))
        tts_config.t2s_weights_path = self.get_t2s_weights_path()
        tts_config.vits_weights_path = self.get_vits_weights_path()
        return tts_config


class TTS_Inference_Run_Input:
    def __init__(self, text: str, text_lang: str = "all_zh"):
        self.text = text
        self.text_lang = text_lang

class TTS_Run_Input: 
    def __init__(self, text: str, text_lang: str = "all_zh", ref_audio_path: str = None, aux_ref_audio_paths: list = None,
                 prompt_text: str = None, prompt_lang: str = "all_zh", top_k: int = 5, top_p: float = 1.0,
                 temperature: float = 1.0, text_split_method: str = "cut1", batch_size: int = 20,
                 speed_factor: float = 1.0, split_bucket: bool = False, return_fragment: bool = False,
                 fragment_interval: float = 0.3, seed: int = -1, parallel_infer: bool = True,
                 repetition_penalty: float = 1.35, sample_steps: int = 32, super_sampling: bool = False):
        self.text = text
        self.text_lang = text_lang
        self.ref_audio_path = ref_audio_path
        self.aux_ref_audio_paths = aux_ref_audio_paths or []
        self.prompt_text = prompt_text
        self.prompt_lang = prompt_lang
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.text_split_method = text_split_method
        self.batch_size = batch_size
        self.speed_factor = speed_factor
        self.split_bucket = split_bucket
        self.return_fragment = return_fragment
        self.fragment_interval = fragment_interval
        self.seed = seed
        self.parallel_infer = parallel_infer
        self.repetition_penalty = repetition_penalty
        self.sample_steps = sample_steps
        self.super_sampling = super_sampling

    @staticmethod
    def get_default_input(
        text: str = "你好，王岚。今天的天气怎么样？",
        text_lang: str = "all_zh",
        ref_audio_path: str = None,
        prompt_text: str = None,
        prompt_lang: str = "all_zh",
    ):
        return TTS_Run_Input(
            text=text,
            text_lang=text_lang,
            ref_audio_path=ref_audio_path,
            aux_ref_audio_paths=[],
            prompt_text=prompt_text,
            prompt_lang=prompt_lang,
            top_k=5,
            top_p=1.0,
            temperature=1.0,
            text_split_method="cut1",
            batch_size=20,
            speed_factor=1.0,
            split_bucket=False,
            return_fragment=False,
            fragment_interval=0.3,
            seed=-1,
            parallel_infer=True,
            repetition_penalty=1.35,
            sample_steps=32,
            super_sampling=False,
        )

class TTS_Inference:
    def __init__(self, model_package_path: str):
        self.model_package_config = TTS_Model_Package(model_package_path)
        self.tts_config = self.model_package_config.get_tts_config()
        self.tts_pipeline = TTS(self.tts_config)

    def run(self, input : TTS_Inference_Run_Input):
        # Convert the strongly-typed input to a dictionary
        run_input = TTS_Run_Input.get_default_input(
            text=input.text,
            text_lang=input.text_lang,
            ref_audio_path=self.model_package_config.get_ref_audio_path(),
            prompt_text=self.model_package_config.get_ref_text_free(),
            prompt_lang=self.model_package_config.get_ref_text_free_lang(),
        )
        return self.tts_pipeline.run(vars(run_input))


if __name__ == "__main__":
    model_package_path: str = "GPT_SoVITS_Inference/model_packages/艾莲"
    tts_inference = TTS_Inference(model_package_path= model_package_path)
    tts_generator = tts_inference.run(TTS_Inference_Run_Input(text= "你好，王岚。今天天气怎么样？"))
    import numpy as np

    sr, audio_data = next(tts_generator)
    sf.write("output.wav", audio_data, sr, format='WAV')

    # Write audio to BytesIO buffer using soundfile
    # For writing to a file, you can write directly without BytesIO:

    # Example: returning audio_bytes in a web framework (pseudo-code)
    # return Response(audio_bytes, media_type="audio/wav")

    # with open("output.wav", "wb") as f:
    #     f.write(audio_data)
    # print("Audio saved to output.wav")

    try:
        tts_generator.close()
    except Exception:
        pass

