# Initialize the whisper model
import mlx_whisper

audio_path = "../data/low_bitrate_ukr.mp3"

result = mlx_whisper.transcribe(
    audio=audio_path,
    verbose=True,
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    word_timestamps=False,
    language="uk",
)

print(result)

def format_time(seconds):
    """ 将秒数格式化为 SRT 时间戳 (小时:分钟:秒,毫秒) """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)

    return f"{hours:02}:{minutes:02}:{sec:02},{ms:03}"

def convert_to_srt(data):
    # Convert the result to srt format
    srt_output = ""
    for idx, segment in enumerate(data["segments"], 1):
        # 格式化开始和结束时间
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])

    # 拼接SRT字幕条目
    srt_output += f"{idx}\n{start_time} --> {end_time}\n{segment['text']}\n\n"

    return srt_output

srt_content = convert_to_srt(result)

print(srt_content)

with open("../data/output.srt", "w") as file:
    file.write(srt_content)
