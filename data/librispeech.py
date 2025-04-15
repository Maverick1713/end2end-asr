import os
import wget
import tarfile
import argparse
import subprocess
from utils import create_manifest
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser(description='Processes and downloads LibriSpeech dataset.')
parser.add_argument("--target-dir", default='LibriSpeech_dataset/', type=str, help="Directory to store the dataset.")
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--files-to-use', default="train-clean-100.tar.gz,"
                                              "train-clean-360.tar.gz,train-other-500.tar.gz,"
                                              "dev-clean.tar.gz,dev-other.tar.gz,"
                                              "test-clean.tar.gz,test-other.tar.gz", type=str,
                    help='list of file names to download')
parser.add_argument('--min-duration', default=1, type=int,
                    help='Prunes training samples shorter than this (in seconds, default 1)')
parser.add_argument('--max-duration', default=15, type=int,
                    help='Prunes training samples longer than this (in seconds, default 15)')
args = parser.parse_args()

LIBRI_SPEECH_URLS = {
    "train": ["http://www.openslr.org/resources/12/train-clean-100.tar.gz"],
    "val": ["http://www.openslr.org/resources/12/dev-clean.tar.gz",
            "http://www.openslr.org/resources/12/dev-other.tar.gz"],
    "test_clean": ["http://www.openslr.org/resources/12/test-clean.tar.gz"],
    "test_other": ["http://www.openslr.org/resources/12/test-other.tar.gz"]
}


def _preprocess_transcript(phrase):
    allowed_chars = "abcdefghijklmnopqrstuvwxyz "
    filtered_phrase = ''.join([char for char in phrase.strip().lower() if char in allowed_chars])
    if len(filtered_phrase) == 0 or len(filtered_phrase) > 100:
        return None
    return filtered_phrase


def _process_file(wav_dir, txt_dir, base_filename, root_dir):
    full_recording_path = os.path.join(root_dir, base_filename)
    if not os.path.exists(full_recording_path):
        return None

    wav_recording_path = os.path.join(wav_dir, base_filename.replace(".flac", ".wav"))
    subprocess.call(["sox {} -r {} -b 16 -c 1 {}".format(full_recording_path, str(args.sample_rate),
                                                         wav_recording_path)], shell=True)

    txt_transcript_path = os.path.join(txt_dir, base_filename.replace(".flac", ".txt"))
    transcript_file = os.path.join(root_dir, "-".join(base_filename.split('-')[:-1]) + ".trans.txt")
    if not os.path.exists(transcript_file):
        if os.path.exists(wav_recording_path):
            os.remove(wav_recording_path)
        return None

    transcriptions = open(transcript_file).read().strip().split("\n")
    transcriptions = {t.split()[0].split("-")[-1]: " ".join(t.split()[1:]) for t in transcriptions}
    key = base_filename.replace(".flac", "").split("-")[-1]
    if key not in transcriptions:
        if os.path.exists(wav_recording_path):
            os.remove(wav_recording_path)
        return None

    transcription = _preprocess_transcript(transcriptions[key])
    if transcription:
        with open(txt_transcript_path, "w") as f:
            f.write(transcription)
        return wav_recording_path, transcription
    else:
        if os.path.exists(wav_recording_path):
            os.remove(wav_recording_path)
        return None


def create_manifest(split_dir, manifest_name, min_duration=None, max_duration=None):
    wav_dir = os.path.join(split_dir, "wav")
    txt_dir = os.path.join(split_dir, "txt")
    manifest_path = os.path.join(split_dir, manifest_name)

    with open(manifest_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["wav_path", "duration", "transcript"])
        for file in os.listdir(wav_dir):
            if not file.endswith(".wav"):
                continue
            wav_path = os.path.join(wav_dir, file)
            txt_path = os.path.join(txt_dir, file.replace(".wav", ".txt"))
            if not os.path.exists(txt_path):
                continue

            duration = float(subprocess.check_output(["soxi -D {}".format(wav_path)], shell=True).strip())
            if min_duration and duration < min_duration:
                continue
            if max_duration and duration > max_duration:
                continue

            with open(txt_path, "r") as tf:
                transcript = tf.read().strip()
            writer.writerow([wav_path, duration, transcript])


def main():
    target_dl_dir = args.target_dir
    if not os.path.exists(target_dl_dir):
        os.makedirs(target_dl_dir)

    files_to_dl = args.files_to_use.strip().split(',')
    for split_type, lst_libri_urls in LIBRI_SPEECH_URLS.items():
        split_dir = os.path.join(target_dl_dir, split_type)
        os.makedirs(os.path.join(split_dir, "wav"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "txt"), exist_ok=True)
        extracted_dir = os.path.join(split_dir, "LibriSpeech")

        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)

        for url in lst_libri_urls:
            if not any(f in url for f in files_to_dl):
                print(f"Skipping url: {url}")
                continue

            filename = url.split("/")[-1]
            target_filename = os.path.join(split_dir, filename)
            if not os.path.exists(target_filename):
                wget.download(url, split_dir)

            print(f"\nUnpacking {filename}...")
            tar = tarfile.open(target_filename)
            tar.extractall(split_dir)
            tar.close()
            os.remove(target_filename)

            print("Converting FLAC files to WAV and extracting transcripts...")
            assert os.path.exists(extracted_dir), f"Archive {filename} not properly uncompressed."
            for root, _, files in tqdm(os.walk(extracted_dir)):
                for f in files:
                    if f.endswith(".flac"):
                        _process_file(
                            wav_dir=os.path.join(split_dir, "wav"),
                            txt_dir=os.path.join(split_dir, "txt"),
                            base_filename=f,
                            root_dir=root
                        )
            print(f"Finished processing {url}")
            shutil.rmtree(extracted_dir)

        if split_type == 'train':
            create_manifest(split_dir, 'libri_train_manifest.csv', args.min_duration, args.max_duration)
        else:
            create_manifest(split_dir, f'libri_{split_type}_manifest.csv')


if __name__ == "__main__":
    main()