import os
import argparse
import tarfile
import shutil
import subprocess
import wget
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser(description='Processes and downloads LibriSpeech dataset.')
parser.add_argument("--target-dir", default='LibriSpeech_dataset/', type=str, help="Directory to store the dataset.")
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--files-to-use', default="train-clean-100.tar.gz,"
                                              "train-clean-360.tar.gz,train-other-500.tar.gz,"
                                              "dev-clean.tar.gz,dev-other.tar.gz,"
                                              "test-clean.tar.gz,test-other.tar.gz", type=str,
                    help='list of file names to download')
parser.add_argument('--min-duration', default=1, type=int, help='Minimum clip duration (seconds)')
parser.add_argument('--max-duration', default=15, type=int, help='Maximum clip duration (seconds)')
args = parser.parse_args()

LIBRI_SPEECH_URLS = {
    "train": ["http://www.openslr.org/resources/12/train-clean-100.tar.gz"],
    "val": ["http://www.openslr.org/resources/12/dev-clean.tar.gz", "http://www.openslr.org/resources/12/dev-other.tar.gz"],
    "test": ["http://www.openslr.org/resources/12/test-clean.tar.gz", "http://www.openslr.org/resources/12/test-other.tar.gz"]
}

BASE_SAVE_DIR = "/content/end2end-asr/data"

def _preprocess_transcript(phrase):
    allowed_chars = "abcdefghijklmnopqrstuvwxyz "
    filtered = ''.join([c for c in phrase.strip().lower() if c in allowed_chars])
    return filtered if 0 < len(filtered) else None

def _process_file(wav_dir, txt_dir, base_filename, root_dir):
    full_path = os.path.join(root_dir, base_filename)
    wav_path = os.path.join(wav_dir, base_filename.replace(".flac", ".wav"))
    subprocess.call(f"sox {full_path} -r {args.sample_rate} -b 16 -c 1 {wav_path}", shell=True)

    txt_path = os.path.join(txt_dir, base_filename.replace(".flac", ".txt"))
    transcript_file = os.path.join(root_dir, "-".join(base_filename.split('-')[:-1]) + ".trans.txt")
    transcriptions = {line.split()[0].split("-")[-1]: " ".join(line.split()[1:]) 
                      for line in open(transcript_file).read().strip().split("\n")}
    key = base_filename.replace(".flac", "").split("-")[-1]
    transcript = _preprocess_transcript(transcriptions.get(key, ""))
    if transcript:
        with open(txt_path, "w") as f:
            f.write(transcript)
            f.flush()

def create_manifest(split_dir, manifest_filename, min_dur=None, max_dur=None):
    wav_dir = os.path.join(split_dir, "wav")
    txt_dir = os.path.join(split_dir, "txt")
    entries = []
    for fname in os.listdir(wav_dir):
        if not fname.endswith(".wav"): continue
        wav_path = os.path.join(wav_dir, fname)
        txt_path = os.path.join(txt_dir, fname.replace(".wav", ".txt"))
        if not os.path.exists(txt_path): continue
        dur = float(subprocess.check_output(f"soxi -D {wav_path}", shell=True).decode().strip())
        if min_dur and dur < min_dur: continue
        if max_dur and dur > max_dur: continue
        transcript = open(txt_path).read().strip()
        abs_wav_path = os.path.join(BASE_SAVE_DIR, os.path.relpath(wav_path, '.'))
        entries.append([abs_wav_path, dur, transcript])

    df = pd.DataFrame(entries)
    df.to_csv(os.path.join(BASE_SAVE_DIR, "LibriSpeech_dataset", manifest_filename), sep='\t', index=False, header=False)

def main():
    os.makedirs(args.target_dir, exist_ok=True)
    files_to_dl = args.files_to_use.strip().split(',')
    for split, urls in LIBRI_SPEECH_URLS.items():
        split_dir = os.path.join(args.target_dir, split)
        wav_dir = os.path.join(split_dir, "wav")
        txt_dir = os.path.join(split_dir, "txt")
        os.makedirs(wav_dir, exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)
        extracted_dir = os.path.join(split_dir, "LibriSpeech")
        if os.path.exists(extracted_dir): shutil.rmtree(extracted_dir)
        for url in urls:
            if not any(f in url for f in files_to_dl):
                print(f"Skipping {url}")
                continue
            filename = url.split("/")[-1]
            dest_file = os.path.join(split_dir, filename)
            if not os.path.exists(dest_file): wget.download(url, split_dir)
            with tarfile.open(dest_file) as tar:
                tar.extractall(split_dir)
            os.remove(dest_file)
            for root, _, files in tqdm(os.walk(extracted_dir)):
                for f in files:
                    if f.endswith(".flac"):
                        _process_file(wav_dir, txt_dir, f, root)
            shutil.rmtree(extracted_dir)

        manifest_name = f"{split}_manifest.csv"
        create_manifest(split_dir, manifest_name, args.min_duration if split == 'train' else None,
                        args.max_duration if split == 'train' else None)

if __name__ == "__main__":
    main()
