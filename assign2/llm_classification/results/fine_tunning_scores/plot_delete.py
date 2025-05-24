import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Create the data
data = {
    'model': ['bert', 'roberta', 'bert', 'roberta', 'bert', 'roberta', 'bert', 'roberta'],
    'training_method': [None, None, 'domain_adaptation', 'domain_adaptation', 'lora', 'lora', 'ia3', 'ia3'],
    'f1_score': [0.9178, 0.9187, 0.92117, 0.91956, 0.89717, 0.90118, 0.7726, 0.85116],
    'training_time': [10309, 9813, 11002, 10926, 7612, 7327, 7626, 7417]
}
df = pd.DataFrame(data)
df['training_method'] = df['training_method'].fillna('base')

plt.figure(figsize=(14, 9))
sns.set_style("whitegrid")

# define your color palette
base_palette = sns.dark_palette("#69d", reverse=True, n_colors=2)
model_colors = {'bert': base_palette[-1], 'roberta': base_palette[0]}

# plot points
for model in ['bert','roberta']:
    subset = df[df['model']==model]
    plt.scatter(subset['training_time'],
                subset['f1_score'],
                c=[model_colors[model]],
                marker='o',
                s=100,
                alpha=0.85,
                edgecolors='black',
                linewidth=1,
                label=model.upper())

# annotate
for _, row in df.iterrows():
    plt.annotate(
        row['training_method'].replace('_', ' ').title(),
        (row['training_time'], row['f1_score']),
        xytext=(10, 0), textcoords='offset points',
        fontsize=9, alpha=0.85,
        ha='left', va='center'
    )

# **NEW: expand the x-axis**
ax = plt.gca()
xmin, xmax = ax.get_xlim()
ax.set_xlim(xmin, xmax + 500)

plt.xlabel('Training Time (seconds)', fontsize=13)
plt.ylabel('F1-Score', fontsize=13)
plt.title('Model Performance: F1-Score vs. Training Time Using Different Training Methods',
          fontsize=16, pad=20)

# legend
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
           title="Model", fontsize=10, title_fontsize=12)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(rect=[0, 0, 0.85, 1])

plt.savefig("training_time_vs_f1_scatter_padded.png", dpi=300, bbox_inches='tight')
plt.show()
