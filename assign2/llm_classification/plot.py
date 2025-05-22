import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

df = pd.read_csv("combined_f1_scores.csv")

df_filtered = df[(df['dataset'] == 'Intel/polite-guard') & (df['split'] == 'test')]

df_agg = df_filtered.groupby(['model', 'prompt_type'], as_index=False)['f1_score'].mean()

pivot_df = df_agg.pivot(index='model', columns='prompt_type', values='f1_score')
pivot_df = pivot_df[['Zero-Shot', 'Few-Shot', 'FS-CoT']]

prompt_types = pivot_df.columns.tolist()
colors = sns.dark_palette("#69d", reverse=True, n_colors=len(prompt_types))

ax = pivot_df.plot(kind='bar', figsize=(10, 6), color=colors)
ax.set_xlabel('Model')
ax.set_ylabel('Weighted F1 Score')
ax.set_title("LLM Comparison on Intel/polite-guard (800 samples from Test set)")
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig("llm_comparison_plot.png", dpi=300)

plt.show()
