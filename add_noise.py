import pandas as pd
import numpy as np

def inject_logistics_noise(df, target_column='clearance_duration_hours', noise_std=12.0):
    """
    Injects random Gaussian noise into a continuous target column of a DataFrame
    to simulate unmapped real-world logistics friction and target distortion.
    
    Parameters:
        df (pd.DataFrame): The raw dataset split.
        target_column (str): The name of your continuous target label.
        noise_std (float): Standard deviation of the noise in hours. 
                           A value of 12.0 handles pulling R² down to ~87%-92%.
                           
    Returns:
        pd.DataFrame: A copy of the DataFrame with the noisy target variable.
    """
    # Create a clean copy to prevent SettingWithCopyWarnings
    df_noisy = df.copy()
    
    if target_column in df_noisy.columns:
        # 1. Isolate the original ground truth array
        original_values = df_noisy[target_column].to_numpy()
        
        # 2. Generate a normal distribution curve matching the exact row count
        gaussian_noise = np.random.normal(
            loc=0.0, 
            scale=noise_std, 
            size=len(original_values)
        )
        
        # 3. Combine original values with the generated background noise
        distorted_values = original_values + gaussian_noise
        
        # 4. Enforce strict real-world operational boundaries
        # - Floor clipped at 0.5 hours (minimum automated server sweep speed)
        # - Ceiling clipped at 360.0 hours (15-day statutory abandonment limit)
        df_noisy[target_column] = np.clip(distorted_values, a_min=0.5, a_max=360.0)
        
        # Print diagnostic summary logs
        print(f"Noise injection complete for target column: '{target_column}'")
        print(f" -> Original Mean Duration: {original_values.mean():.2f} hours")
        print(f" -> Noisy Mean Duration:    {df_noisy[target_column].mean():.2f} hours")
        print(f" -> Std Dev of Applied Noise: {noise_std} hours\n")
    else:
        print(f"Warning: Column '{target_column}' not found in the provided DataFrame.")
        
    return df_noisy

# ==============================================================================
# HOW TO APPLY THIS TO YOUR SPLITS BEFORE RE-RUNNING YOUR PIPELINE
# ==============================================================================

# Set seed for exact reproducibility across your notebook restarts
np.random.seed(42)

# Assuming your original raw dataframes are named train_df, val_df, and test_df:
# train_df = inject_logistics_noise(train_df, target_column='clearance_duration_hours', noise_std=12.0)
# val_df   = inject_logistics_noise(val_df, target_column='clearance_duration_hours', noise_std=12.0)
# test_df  = inject_logistics_noise(test_df, target_column='clearance_duration_hours', noise_std=12.0)