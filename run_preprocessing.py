"""
Convenience script for applying the Youth Risk Behavior Survey (2024) data
preprocessing pipeline to a CSV file.

Usage
-----
Adjust the ``input_path``, ``output_path`` and ``nominal_cols`` variables
below to match your raw data file and the set of variables that should be
treated as nominal (non‑ordered) categories.  Run the script with

```
python run_preprocessing.py
```

The processed CSV will be written to ``output_path``.
"""

import pandas as pd
from preprocess_youth_data import preprocess_youth_data


def main() -> None:
    # TODO: Update these paths to point to your raw and output files.  The
    # encoding of the raw CSV may need to be specified (e.g., 'euc-kr' for
    # Korean encoded files).
    input_path = 'adolescent_health_data_1.csv'
    output_path = 'KYRBS_2024_processed.csv'

    # Specify columns whose numeric codes do *not* reflect an inherent order.
    # These variables will be one‑hot encoded.  Examples include:
    #   - SEX (1=남자, 2=여자)
    #   - E_RES (현재 거주형태: 가족과 함께, 한부모, 친척과 등)
    # Add additional columns as needed based on the survey documentation.
    nominal_cols = ['SEX', 'E_RES']

    print(f'Reading raw data from {input_path}...')
    df_raw = pd.read_csv(input_path ) #, encoding='euc-kr'
    print('Raw data loaded.')

    print('Running preprocessing...')
    df_processed = preprocess_youth_data(df_raw, nominal_cols=nominal_cols)
    print('Preprocessing complete.')

    print(f'Writing processed data to {output_path}...')
    df_processed.to_csv(output_path, index=False)
    print('Done.')


if __name__ == '__main__':
    main()