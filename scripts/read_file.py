import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logging.info("Start reading xlsx file")
lists_of_sheets = ["UX", "GP", "AS"]
input_file_path = Path(
    r"C:\Users\iliam\Desktop\My first NLP project\data\raw\data_raw.xlsx"
)
df = pd.read_excel(input_file_path, "UX")
output_file_path = Path(
    r"C:\Users\iliam\Desktop\My first NLP project\data\interim\data_read.csv"
)
full_df = pd.DataFrame(columns=df.columns)
for sheet in tqdm(lists_of_sheets, desc="Reading sheets"):
    df = pd.read_excel(input_file_path, sheet)
    full_df = pd.concat([full_df, df], ignore_index=True)
full_df.to_csv(output_file_path, index=False)
logging.info("Done reading xlsx file and saved to csv")
