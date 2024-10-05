# whisper-turbo-local-benchmarks

This repository is a fast and turbo attempt to assess the accuracy of the new Whisper turbo model just released against previous versions. You can use this scripts with your own files, 100% locally.

## How to use it

First create a virtual environment. I do use uv for this:
```bash
uv venv .venv
source .venv/bin/activate
```

Install the required packages:
```bash
uv pip install whisper openai-whisper matplotlib seaborn pandas tqdm jinja2 nltk rouge-score bert-score python-Levenshtein
```

Put your audio files in input folder and then run the following command:
```bash
python run_whisper_benchmarks.py
```

To produce the results execute the following command:
```bash
python analyze_whisper_benchmark_results.py
```

## Notes

I am not quite convinced about the output metrics at the moment, besides the time it takes to process the audio files. But it's useful to generate the output of the different whisper models and visually compare the outputs using diff in your favorite IDE :)