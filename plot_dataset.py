import matplotlib.pyplot as plt
import pandas as pd


def main():
  print('Reading ml_training_dataset.csv...')
  try:
    df = pd.read_csv('ml_training_dataset.csv')
  except FileNotFoundError:
    print(
        "Error: 'ml_training_dataset.csv' not found. Please run the dataset"
        ' generator first.'
    )
    return

  print(f'Loaded dataset with {len(df)} records.')

  # Optional sampling for heavy datasets to keep plots responsive
  if len(df) > 5000:
    df_plot = df.sample(n=5000, random_state=42)
  else:
    df_plot = df

  # --- 3D Scatter Plot: P0 vs Duration vs P1 ---
  fig = plt.figure(figsize=(12, 9))
  ax = fig.add_subplot(projection='3d')

  x = df_plot['initial_price_p0']
  y = df_plot['duration']
  z = df_plot['target_price_p1']

  # Color the points based on P1 to easily spot high/low future prices
  scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=12, alpha=0.6)

  ax.set_xlabel('Initial Price (P0)')
  ax.set_ylabel('Duration (Years)')
  ax.set_zlabel('Target Future Price (P1)')
  ax.set_title('3D Market Space: Initial Price vs Duration vs Future Price')

  fig.colorbar(scatter, label='Target Price P1')
  plt.savefig('market_space_3d.png', dpi=300, bbox_inches='tight')
  print("Saved clean 3D plot as 'market_space_3d.png'")
  plt.show()


if __name__ == '__main__':
  main()