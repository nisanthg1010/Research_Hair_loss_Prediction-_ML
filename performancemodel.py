import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation
data = {
    'Model Version': [
        'Base Model (ExtraTrees - GridSearch)', 'Calibrated Classifier (ET - Isotonic)',
        'Voting Ensemble (ET + RF + GB)', 'Ensemble Variant 1', 'Ensemble Variant 2',
        'Best Ensemble (Final Reported)'
    ],
    'Accuracy (%)': [85.00, 92.17, 91.67, 91.83, 91.00, 92.17],
    'Precision (0)': [0.90, 0.94, 0.94, 0.95, 0.96, 0.95],
    'Precision (1)': [0.74, 0.89, 0.89, 0.88, 0.86, 0.88],
    'Recall (0)': [0.89, 0.93, 0.92, 0.91, 0.90, 0.91],
    'Recall (1)': [0.75, 0.91, 0.91, 0.93, 0.94, 0.93],
    'F1-Score (0)': [0.89, 0.93, 0.93, 0.93, 0.93, 0.93],
    'F1-Score (1)': [0.75, 0.90, 0.90, 0.90, 0.90, 0.91]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set dark background for plotting
plt.style.use('dark_background')

# Create stacked bar chart
df_stacked = df.set_index('Model Version')[['Precision (0)', 'Precision (1)', 'Recall (0)', 'Recall (1)', 'F1-Score (0)', 'F1-Score (1)']]
ax = df_stacked.plot(kind='bar', stacked=True, figsize=(12,6), color=sns.color_palette("inferno", len(df_stacked.columns)))

# Add title and labels
plt.title('Stacked Bar Chart - Model Performance', fontsize=16, color='white')
plt.ylabel('Metric Value', color='white')
plt.xlabel('Model Version', color='white')
plt.xticks(rotation=45, ha='right', color='white')
plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='white', fontsize=10)

# Add the values on top of each bar
for p in ax.patches:
    height = p.get_height()
    if height > 0:  # To avoid plotting values for empty bars
        ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., p.get_y() + height / 2.),
                    xytext=(0, 0), textcoords='offset points', ha='center', va='center',
                    color='white', fontsize=10)

# Adjust layout to avoid clipping
plt.tight_layout()

# Show the plot
plt.show()
