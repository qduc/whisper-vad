import argparse
import json
import os
import torch
import moviepy.editor as mp
import webvtt
import time
from utils import add_timestamps, convert_wav_offset_to_timestamp

SAMPLING_RATE = 16000

def extract_audio_from_video(video_path: str, audio_path: str):
    if os.path.exists(audio_path):
        return  # Skip if audio file already exists
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

    wav = read_audio(audio_path, sampling_rate=SAMPLING_RATE)
    # get speech timestamps from full audio file
    torch.set_num_threads(1)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
    print(speech_timestamps)

    for i, segment in enumerate(speech_timestamps):
        # print(f"Segment {i + 1}: {segment['start']} - {segment['end']}")
        save_audio(f"{output_folder}/segment_{i + 1}.wav", collect_chunks([segment], wav), SAMPLING_RATE)

    return speech_timestamps

def translate_audio_segments(segments, output_folder, whisper_exec):
    files = ' '.join(f"{output_folder}/segment_{i}.wav" for i in range(1, len(segments) + 1))
    with open(os.devnull, 'w') as devnull:
        os.system(f"{whisper_exec} {files}")

def merge_subtitle(segments, subtitle_folder):
    final_timestamps = []
    for i, segment in enumerate(segments):
        path = f"{subtitle_folder}/segment_{i + 1}.wav.vtt"
        captions = webvtt.read(path)
        for caption in captions:
            start_time_offset = caption.start
            start_time = add_timestamps(convert_wav_offset_to_timestamp(segment['start'], SAMPLING_RATE), start_time_offset)
            end_time_offset = caption.end
            end_time = add_timestamps(convert_wav_offset_to_timestamp(segment['start'], SAMPLING_RATE), end_time_offset)
            text = caption.text
            final_timestamps.append({
                'start': start_time,
                'end': end_time,
                'text': text
            })
            # print(f"[{start_time} --> {end_time}]  {text}")
    # print(final_timestamps)
    return final_timestamps

def write_subtitle(subtitles, output_path):
    with open(output_path, 'w') as f:
        for i, subtitle in enumerate(subtitles):
            f.write(f"{i + 1}\n")
            f.write(f"{subtitle['start']} --> {subtitle['end']}\n")
            f.write(f"{subtitle['text']}\n")
            f.write("\n")

def arg_parser():
    parser = argparse.ArgumentParser(description="Transcribe video to text.")

    # Add arguments
    parser.add_argument('-m', '--model', type=str, default='medium',
                        help='The model size. For example: small, medium, or large.')
    parser.add_argument('-l', '--language', type=str, required=True,
                        help='The language. For example: en, ja, etc.')
    parser.add_argument('file', type=str,
                        help='The input file path.')

    # Parse the arguments
    args = parser.parse_args()

    return args

def get_config():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            return config
    except IOError as e:
        print('Config file not found. Please create a config.json file from config.example.json.')
        exit()

def main():
    args = arg_parser()
    config = get_config()

    video_path = args.file  # Change this to your video's path
    filename = os.path.splitext(os.path.basename(video_path))[0]  # The filename of the video
    audio_path = f'{filename}.wav'  # The path to save the extracted audio
    output_folder = 'audio_chunks'  # Folder to save audio chunks
    whisper_dir = config['whisper.cpp_dir']  # The whisper directory
    whisper_model = args.model  # The whisper model to use
    language = args.language  # The language of video
    whisper_exec = f'{whisper_dir}/main -m {whisper_dir}/models/ggml-{whisper_model}.bin -ovtt -l {language} -tr'

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

    vad_end = time.time()
    vad_duration = vad_end - start

    # Translate with whisper
    translate_audio_segments(segments, output_folder, whisper_exec)

    translate_end = time.time()
    translate_duration = translate_end - vad_end

    subtitles = merge_subtitle(segments, output_folder)

    write_subtitle(subtitles, f'{filename}.srt')

    os.remove(audio_path)
    for file in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, file))

    print(f"Finished. Time taken: {time.time() - start:.2f} seconds")
    print(f"VAD duration: {vad_duration:.2f} seconds")
    print(f"Translate duration: {translate_duration:.2f} seconds")

if __name__ == '__main__':
    main()