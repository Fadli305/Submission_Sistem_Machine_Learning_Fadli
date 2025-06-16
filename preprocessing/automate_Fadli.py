import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_personality_dataset(input_path, output_path):
    """
    Melakukan preprocessing dataset personality:
    - Label Encoding untuk fitur kategorikal
    - Standardisasi fitur numerik
    - Simpan hasil ke output_path
    """
    # Load dataset
    df = pd.read_csv(input_path)

    # Encode fitur kategorikal biner
    le = LabelEncoder()
    df['Stage_fear'] = le.fit_transform(df['Stage_fear'])  # Yes=1, No=0
    df['Drained_after_socializing'] = le.fit_transform(df['Drained_after_socializing'])

    # Encode label target
    df['Personality'] = le.fit_transform(df['Personality'])  # Extrovert=0, Introvert=1

    # Identifikasi fitur numerik (exclude target)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('Personality')

    # Standarisasi fitur numerik
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Simpan hasil
    df.to_csv(output_path, index=False)
    print(f"Preprocessing selesai. File disimpan di: {output_path}")


# Contoh pemanggilan fungsi
if __name__ == "__main__":
    preprocess_personality_dataset(
        input_path='dataset/personality_dataset.csv',
        output_path='preprocessing/personality_dataset_preprocessing.csv'
    )
