import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation based on your provided results
data = {
    'Model Version': [
        'Decision Tree', 
        'Extra Trees', 
        'XGBoost', 
        'Ensemble Model 1 (RF+KNN+GB)', 
        'Ensemble Model 2 (XG+LG+RF)'
    ],
    'Accuracy (%)': [85.00, 92.17, 91.67, 91.17, 93.17],
    'Precision (0)': [0.90, 0.94, 0.94, 0.96, 0.95],
    'Precision (1)': [0.74, 0.89, 0.88, 0.86, 0.90],
    'Recall (0)': [0.89, 0.93, 0.92, 0.89, 0.93],
    'Recall (1)': [0.75, 0.91, 0.91, 0.94, 0.93],
    'F1-Score (0)': [0.89, 0.93, 0.93, 0.92, 0.94],
    'F1-Score (1)': [0.75, 0.90, 0.90, 0.89, 0.92]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set dark background for plotting
plt.style.use('dark_background')

# Create stacked bar chart
df_stacked = df.set_index('Model Version')[['Precision (0)', 'Precision (1)', 'Recall (0)', 'Recall (1)', 'F1-Score (0)', 'F1-Score (1)']]
ax = df_stacked.plot(
    kind='bar', 
    stacked=True, 
    figsize=(14, 7), 
    color=sns.color_palette("inferno", len(df_stacked.columns))
)

# Add title and labels
plt.title('Model Performance Comparison - Stacked Bar Chart', fontsize=18, color='white')
plt.ylabel('Metric Value', color='white', fontsize=12)
plt.xlabel('Model Version', color='white', fontsize=12)
plt.xticks(rotation=30, ha='right', color='white', fontsize=10)
plt.yticks(color='white')
plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='white', fontsize=10, title_fontsize=12)

# Add values inside each bar
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f'{height:.2f}', 
            (p.get_x() + p.get_width() / 2., p.get_y() + height / 2.), 
            xytext=(0, 0), 
            textcoords='offset points', 
            ha='center', 
            va='center',
            color='white',
            fontsize=8
        )

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

# Print the performance table nicely
print("\nModel Performance Table:\n")
print(df.to_string(index=False))
