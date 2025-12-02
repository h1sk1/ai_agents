# 3️⃣ Initialize a pipeline
import gc

from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import sounddevice as sd
import numpy as np
# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
pipeline = KPipeline(lang_code='a') # <= make sure lang_code matches voice

# This text is for demonstration purposes only, unseen during training
text = [
'''
Try saying it slowly: [im](/ɪm/) . . . . . [por](/pˈɔɹ/) . . . . . [tant](/tᵊnt/) . . . . .
''',
'''
Can you read it out loud, and send it back to me?
'''
]

# 4️⃣ Generate, display, and save audio files in a loop.
generator = pipeline(
    text, voice='af_heart', # <= change voice here
    speed=0.8, split_pattern=r'\n+'
)

sample_rate = 24000
channels = 1

# 使用 with 语句管理音频流
with sd.OutputStream(samplerate=sample_rate, channels=channels, dtype='float32') as stream:
    for i, (gs, ps, audio) in enumerate(generator):
        try:
            # 确保音频格式正确
            audio = np.array(audio, dtype=np.float32)
            print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")

            # 检查声道数是否需要调整
            if audio.ndim == 2 and audio.shape[1] > 1:
                channels = audio.shape[1]
                stream.channels = channels

            # 写入音频
            stream.write(audio)
        except Exception as e:
            print(f"写入音频时出错: {e}")
            break
