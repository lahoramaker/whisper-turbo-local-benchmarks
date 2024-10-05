import os
import json
import whisper
from difflib import SequenceMatcher
from tqdm import tqdm
import time
import nltk
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import Levenshtein

# Download NLTK data for sentence tokenization
nltk.download('punkt', quiet=True)

def transcribe_audio(model, audio_path):
    start_time = time.time()
    result = model.transcribe(audio_path)
    end_time = time.time()
    return result["text"], end_time - start_time

def calculate_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

def calculate_rouge(text1, text2):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(text1, text2)
    return {
        "rouge-1": scores['rouge1'].fmeasure,
        "rouge-2": scores['rouge2'].fmeasure,
        "rouge-l": scores['rougeL'].fmeasure
    }

def calculate_bert_score(text1, text2):
    P, R, F1 = bert_score([text1], [text2], lang="en", verbose=False)
    return F1.item()

def calculate_levenshtein_distance(text1, text2):
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)
    distances = [Levenshtein.distance(s1, s2) for s1, s2 in zip(sentences1, sentences2)]
    return sum(distances) / len(distances) if distances else 0

def benchmark_models(input_folder, output_folder):
    models = {
        "large": whisper.load_model("large"),
        "medium": whisper.load_model("medium"),
        "turbo": whisper.load_model("turbo")
    }

    results = []

    for audio_file in tqdm(os.listdir(input_folder)):
        if audio_file.endswith(('.mp3', '.wav', '.flac')):
            audio_path = os.path.join(input_folder, audio_file)
            
            transcriptions = {}
            times = {}
            for model_name, model in models.items():
                transcription, transcription_time = transcribe_audio(model, audio_path)
                transcriptions[model_name] = transcription
                times[model_name] = transcription_time
                
                # Save transcription to file
                with open(os.path.join(output_folder, f"{audio_file}_{model_name}.txt"), "w") as f:
                    f.write(transcription)

            large_transcription = transcriptions["large"]
            result = {
                "audio": audio_file,
                "transcriptions": transcriptions,
                "times": times,
                "metrics": {
                    "large": {
                        "similarity": 1.0,
                        "rouge": calculate_rouge(large_transcription, large_transcription),
                        "bert_score": 1.0,
                        "levenshtein": 0
                    }
                }
            }

            for model in ["medium", "turbo"]:
                similarity = calculate_similarity(large_transcription, transcriptions[model])
                rouge_scores = calculate_rouge(large_transcription, transcriptions[model])
                bert_score = calculate_bert_score(large_transcription, transcriptions[model])
                levenshtein = calculate_levenshtein_distance(large_transcription, transcriptions[model])
                
                result["metrics"][model] = {
                    "similarity": similarity,
                    "rouge": rouge_scores,
                    "bert_score": bert_score,
                    "levenshtein": levenshtein
                }

            results.append(result)

            print(f"Audio: {audio_file}")
            for model in models.keys():
                print(f"{model.capitalize()} transcription (first 100 chars): {transcriptions[model][:100]}...")
                print(f"{model.capitalize()} transcription time: {times[model]:.2f} seconds")
            print("\n" + "="*50 + "\n")

    return results

def save_results(results, filename='benchmark_results.json'):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    input_folder = "./input"
    output_folder = "./output"
    os.makedirs(output_folder, exist_ok=True)
    results = benchmark_models(input_folder, output_folder)
    save_results(results, os.path.join(output_folder, 'benchmark_results.json'))
    print("Benchmark complete. Results saved to 'benchmark_results.json'. Check the console output and the generated transcriptions in the output folder.")
    print("Run the analyze_results.py script to generate a detailed HTML report.")