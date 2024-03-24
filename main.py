import os
from datetime import datetime, timedelta
import torch
import moviepy.editor as mp
import webvtt
import time

def extract_audio_from_video(video_path: str, audio_path: str):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)


def get_voice_activity_segments(audio_path: str, output_folder: str = 'audio_chunks'):
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)

    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    sampling_rate = 16000  # also accepts 8000
    wav = read_audio(audio_path, sampling_rate=sampling_rate)
    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
    print(speech_timestamps)

    for i, segment in enumerate(speech_timestamps):
        # print(f"Segment {i + 1}: {segment['start']} - {segment['end']}")
        save_audio(f"{output_folder}/segment_{i + 1}.wav", collect_chunks([segment], wav), 16000)

    return speech_timestamps

def translate_audio_segments(segments, output_folder, whisper_exec):
    files = ' '.join(f"{output_folder}/segment_{i}.wav" for i in range(1, len(segments) + 1))
    with open(os.devnull, 'w') as devnull:
        os.system(f"{whisper_exec} {files} > {devnull.name} 2>&1")

def merge_subtitle(segments):
    def add_timestamps(timestamp1, timestamp2):
        # Parse the timestamps into datetime objects
        dt1 = datetime.strptime(timestamp1, "%H:%M:%S.%f")
        dt2 = datetime.strptime(timestamp2, "%H:%M:%S.%f")

        # Convert the datetime objects to timedelta
        td1 = timedelta(hours=dt1.hour, minutes=dt1.minute, seconds=dt1.second, microseconds=dt1.microsecond)
        td2 = timedelta(hours=dt2.hour, minutes=dt2.minute, seconds=dt2.second, microseconds=dt2.microsecond)

        # Add the timedeltas
        td_sum = td1 + td2

        # Convert the timedelta back into a datetime object
        dt_sum = datetime(1, 1, 1) + td_sum

        # Format the new datetime back into a string
        new_timestamp = dt_sum.strftime("%H:%M:%S.%f")[:-3]  # remove the last 3 digits of microseconds

        return new_timestamp

    def convert_wav_offset_to_timestamp(wav_offset, sampling_rate):
        duration_seconds = wav_offset / sampling_rate
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        seconds = duration_seconds % 60
        return f"{int(hours):02}:{int(minutes):02}:{seconds:.3f}"


    final_timestamps = []
    for i, segment in enumerate(segments):
        path = f"{output_folder}/segment_{i + 1}.wav.vtt"
        captions = webvtt.read(path)
        for caption in captions:
            start_time_offset = caption.start
            start_time = add_timestamps(convert_wav_offset_to_timestamp(segment['start'], 16000), start_time_offset)
            end_time_offset = caption.end
            end_time = add_timestamps(convert_wav_offset_to_timestamp(segment['start'], 16000), end_time_offset)
            text = caption.text
            final_timestamps.append({
                'start': start_time,
                'end': end_time,
                'text': text
            })
            print(f"[{start_time} --> {end_time}]  {text}")
    # print(final_timestamps)
    return final_timestamps

def write_subtitle(subtitles, output_path):
    with open(output_path, 'w') as f:
        for i, subtitle in enumerate(subtitles):
            f.write(f"{i + 1}\n")
            f.write(f"{subtitle['start']} --> {subtitle['end']}\n")
            f.write(f"{subtitle['text']}\n")
            f.write("\n")

video_path = 'demo.mp4'  # Change this to your video's path
audio_path = 'extracted_audio.wav'  # The path to save the extracted audio
output_folder = 'audio_chunks'  # Folder to save audio chunks
whisper_dir = 'D:/src/whisper-cublas'  # The whisper directory
whisper_model = 'medium'  # The whisper model to use
language = 'ko'  # The language of video
whisper_exec = f'{whisper_dir}\main.exe -m {whisper_dir}\models\ggml-{whisper_model}.bin -ovtt -l {language} -tr'

# Create output folder if not exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    for file in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, file))

# Extract the audio
extract_audio_from_video(video_path, audio_path)

start = time.time()

# Get voice activity segments
segments = get_voice_activity_segments(audio_path, output_folder)

# Trasnlate with whisper
translate_audio_segments(segments, output_folder, whisper_exec)

subtitles = merge_subtitle(segments)

write_subtitle(subtitles, 'output.srt')

print(f"Finished. Time taken: {time.time() - start:.2f} seconds")
