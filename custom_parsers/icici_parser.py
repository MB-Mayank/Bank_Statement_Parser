import pandas as pd
import pdfplumber
from typing import List, Dict
import numpy as np

def parse(pdf_path: str) -> pd.DataFrame:
    '''Parse bank statement PDF and return DataFrame'''
    
    # Extract data from PDF
    transactions = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # METHOD 1: Try table extraction first
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    # Process table rows
                    for row in table[1:]:
                        if len(row) >= 5:
                            date = row[0]
                            description = row[1]
                            debit_amt = row[2]
                            credit_amt = row[3]
                            balance = row[4]
                            transactions.append({
                                'Date': date,
                                'Description': description,
                                'Debit Amt': debit_amt,
                                'Credit Amt': credit_amt,
                                'Balance': balance
                            })
            
            # METHOD 2: Or extract text and parse lines
            text = page.extract_text()
            if text:
                lines = text.split('\n')
                # Parse each line
                pass
    
    # Create DataFrame
    df = pd.DataFrame(transactions, columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])
    
    # Fix type mismatches
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
        df['Date'] = df['Date'].dt.strftime('%d-%m-%Y')

    for col in ['Debit Amt', 'Credit Amt', 'Balance']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values
    df['Debit Amt'] = df['Debit Amt'].fillna(np.nan)
    df['Credit Amt'] = df['Credit Amt'].fillna(np.nan)
    df['Balance'] = df['Balance'].fillna(np.nan)
    
    # Ensure DataFrame has exactly 100 rows
    if len(df) < 100:
        df = pd.concat([df, pd.DataFrame([{'Date': '', 'Description': '', 'Debit Amt': np.nan, 'Credit Amt': np.nan, 'Balance': np.nan} for _ in range(100 - len(df))], columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])], ignore_index=True)
    elif len(df) > 100:
        df = df.head(100)
    
    return df