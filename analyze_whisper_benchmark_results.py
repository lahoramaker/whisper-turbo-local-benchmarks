import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import json
import base64
from io import BytesIO

def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_statistics(df):
    stats = df.groupby('model').agg({
        'similarity': ['mean', 'median', 'std', 'min', 'max'],
        'time': ['mean', 'median', 'std', 'min', 'max'],
        'rouge-1': ['mean', 'median', 'std', 'min', 'max'],
        'rouge-2': ['mean', 'median', 'std', 'min', 'max'],
        'rouge-l': ['mean', 'median', 'std', 'min', 'max'],
        'bert_score': ['mean', 'median', 'std', 'min', 'max'],
        'levenshtein': ['mean', 'median', 'std', 'min', 'max']
    }).reset_index()
    stats.columns = ['model', 'similarity_mean', 'similarity_median', 'similarity_std', 'similarity_min', 'similarity_max',
                     'time_mean', 'time_median', 'time_std', 'time_min', 'time_max',
                     'rouge-1_mean', 'rouge-1_median', 'rouge-1_std', 'rouge-1_min', 'rouge-1_max',
                     'rouge-2_mean', 'rouge-2_median', 'rouge-2_std', 'rouge-2_min', 'rouge-2_max',
                     'rouge-l_mean', 'rouge-l_median', 'rouge-l_std', 'rouge-l_min', 'rouge-l_max',
                     'bert_score_mean', 'bert_score_median', 'bert_score_std', 'bert_score_min', 'bert_score_max',
                     'levenshtein_mean', 'levenshtein_median', 'levenshtein_std', 'levenshtein_min', 'levenshtein_max']
    stats = stats.round(4)
    return stats

def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_plots(df):
    plots = {}
    metrics = ['similarity', 'time', 'rouge-1', 'rouge-2', 'rouge-l', 'bert_score', 'levenshtein']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='model', y=metric, data=df)
        plt.title(f'{metric.capitalize()} Comparison')
        plt.ylabel(metric.capitalize())
        plots[f'{metric}_boxplot'] = plot_to_base64(plt.gcf())
        plt.close()

    return plots

def generate_html_report(results, stats, plots):
    template = Template('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Whisper Model Benchmark Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1, h2, h3 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #f2f2f2; }
            img { max-width: 100%; height: auto; margin-bottom: 20px; }
            .plot-container { display: flex; flex-wrap: wrap; justify-content: space-between; }
            .plot-item { width: 48%; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>Whisper Model Benchmark Analysis</h1>
        
        <h2>Statistical Summary</h2>
        {% for metric in ['similarity', 'time', 'rouge-1', 'rouge-2', 'rouge-l', 'bert_score', 'levenshtein'] %}
        <h3>{{ metric.capitalize() }}</h3>
        <table>
            <tr>
                <th>Model</th>
                <th>Mean</th>
                <th>Median</th>
                <th>Std Dev</th>
                <th>Min</th>
                <th>Max</th>
            </tr>
            {% for _, row in stats.iterrows() %}
            <tr>
                <td>{{ row['model'] }}</td>
                <td>{{ row[metric + '_mean'] }}</td>
                <td>{{ row[metric + '_median'] }}</td>
                <td>{{ row[metric + '_std'] }}</td>
                <td>{{ row[metric + '_min'] }}</td>
                <td>{{ row[metric + '_max'] }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endfor %}

        <h2>Performance Visualizations</h2>
        <div class="plot-container">
            {% for metric in ['similarity', 'time', 'rouge-1', 'rouge-2', 'rouge-l', 'bert_score', 'levenshtein'] %}
            <div class="plot-item">
                <h3>{{ metric.capitalize() }} Box Plot</h3>
                <img src="data:image/png;base64,{{ plots[metric + '_boxplot'] }}" alt="{{ metric.capitalize() }} Box Plot">
            </div>
            {% endfor %}
        </div>

        <h2>Detailed Results</h2>
        {% for result in results %}
        <h3>{{ result['audio'] }}</h3>
        {% for model in ['large', 'medium', 'turbo'] %}
        <h4>{{ model.capitalize() }}</h4>
        <p><strong>First paragraph:</strong> {{ result['transcriptions'][model].split('\n')[0] }}</p>
        <p><strong>Transcription time:</strong> {{ result['times'][model] }} seconds</p>
        <p><strong>Metrics:</strong></p>
        <ul>
            <li>Similarity: {{ result['metrics'][model]['similarity'] }}</li>
            <li>ROUGE-1: {{ result['metrics'][model]['rouge']['rouge-1'] }}</li>
            <li>ROUGE-2: {{ result['metrics'][model]['rouge']['rouge-2'] }}</li>
            <li>ROUGE-L: {{ result['metrics'][model]['rouge']['rouge-l'] }}</li>
            <li>BERT Score: {{ result['metrics'][model]['bert_score'] }}</li>
            <li>Levenshtein Distance: {{ result['metrics'][model]['levenshtein'] }}</li>
        </ul>
        {% endfor %}
        <hr>
        {% endfor %}
    </body>
    </html>
    ''')

    html_content = template.render(results=results, stats=stats, plots=plots)
    
    with open('whisper_benchmark_report.html', 'w') as f:
        f.write(html_content)

def main():
    # Load results from the JSON file
    results = load_results('output/benchmark_results.json')
    
    # Convert results to DataFrame
    df = []
    for result in results:
        for model in ['large', 'medium', 'turbo']:
            df.append({
                'audio': result['audio'],
                'model': model,
                'similarity': result['metrics'][model]['similarity'],
                'time': result['times'][model],
                'rouge-1': result['metrics'][model]['rouge']['rouge-1'],
                'rouge-2': result['metrics'][model]['rouge']['rouge-2'],
                'rouge-l': result['metrics'][model]['rouge']['rouge-l'],
                'bert_score': result['metrics'][model]['bert_score'],
                'levenshtein': result['metrics'][model]['levenshtein']
            })
    df = pd.DataFrame(df)
    
    # Calculate statistics
    stats = calculate_statistics(df)
    
    # Create plots
    plots = create_plots(df)
    
    # Generate HTML report
    generate_html_report(results, stats, plots)
    
    print("Analysis complete. Check 'whisper_benchmark_report.html' for the detailed report.")

if __name__ == "__main__":
    main()