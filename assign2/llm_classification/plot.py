import pandas as pd
import matplotlib.pyplot as plt

# Load the combined CSV
df = pd.read_csv("combined_f1_scores.csv")

# Filter for a specific dataset and split if needed
df_filtered = df[(df['dataset'] == 'Intel/polite-guard') & (df['split'] == 'validation')]

# Pivot the data for plotting
pivot_df = df_filtered.pivot(index='model', columns='prompt_type', values='f1_score')

# Plot
ax = pivot_df.plot(kind='bar', figsize=(12, 8))
ax.set_xlabel('Models')
ax.set_ylabel('Weighted F1 Score')
ax.set_title("LLM Comparison: Combined Results")
ax.legend(title='Prompt Types')
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()