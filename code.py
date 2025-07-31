# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 13:38:28 2025

@author: YLI355
"""

# %% Libraries

import pandas as pd 
import numpy as np 
import pdfplumber
import pdb
import os

# %% Functions
def shift_right_if_next_is_nan(row):
    row = row.copy()
    for i in reversed(range(len(row) - 1)):
        if pd.isna(row[i+1]) and not pd.isna(row[i]):
            row[i+1] = row[i]
            row[i] = np.nan
    return row

def shift_row_left(row):
    """Shift non-NaN values to the left within a row."""
    non_nan = [x for x in row if pd.notna(x)]
    return pd.Series(non_nan + [np.nan] * (len(row) - len(non_nan)))

def shift_left(row):
    # Replace blanks with NaN for uniformity
    row = row.replace('', np.nan)
    # Drop NaNs and shift values left, then fill the remaining with NaN
    shifted = row.dropna().reset_index(drop=True)
    # Create a new series of the same length with NaNs
    new_row = pd.Series([np.nan]*len(row))
    # Fill from left with the shifted values
    new_row[:len(shifted)] = shifted
    return new_row

def dropNAs(df):
    """
    Drops columns and rows from a DataFrame that are entirely NaN.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with all-NaN columns and rows removed.
    """
    
    df_result = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    return df_result

def sepxyz(df, XYZ, Xout, Yout, Zout):
    """
    Splits a column containing comma-separated X, Y, Z values into three separate columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        XYZ (str): The name of the column to split (assumes format "X,Y,Z").
        Xout (str): The name for the new column containing X values.
        Yout (str): The name for the new column containing Y values.
        Zout (str): The name for the new column containing Z values.

    Returns:
        pd.DataFrame: The modified DataFrame with X, Y, Z columns extracted and the original column removed.
    """
    
    splits = df[XYZ].str.split(',', n=2, expand=True)

    df[[Xout, Yout, Zout]] = splits.iloc[:, :3]

    df = df.rename(columns={XYZ: 'Drop'})
    df = df.drop(columns='Drop')

    return df

def rename_columns(df_to_rename, rename_df, indices):
    """
    Rename columns in df_to_rename based on rules in rename_df at given indices.
    
    Parameters:
        df_to_rename (pd.DataFrame): The DataFrame whose columns to rename
        rename_df (pd.DataFrame): A DataFrame with a 'rule' column of 'old - new' strings
        indices (list of int): The index positions in rename_df to use for renaming
    
    Returns:
        pd.DataFrame: A copy of the input DataFrame with renamed columns
    """
    rename_map = {}
    for i in indices:
        try:
            rule = rename_df.loc[i, 'rule']
            old, new = map(str.strip, rule.split('-'))
            rename_map[old] = new
        except Exception as e:
            print(f"Skipping index {i} due to error: {e}")
    
    return df_to_rename.rename(columns=rename_map)

def state_abbreviations(dfs, col, new_col):
    """
    Maps USPS state/territory abbreviations in a column to full names.
    
    Parameters:
        dfs (pd.DataFrame or list of pd.DataFrame): DataFrame(s) to process.
        col (str): Name of the column with abbreviations (e.g., 'State').
        new_col (str): Name of the new column to store full state names.
    
    Returns:
        pd.DataFrame or list of pd.DataFrame: Updated DataFrame(s).
    """
    abbreviation_to_name = {
        "AK": "Alaska", "AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona", "CA": "California",
        "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
        "HI": "Hawaii", "IA": "Iowa", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
        "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "MA": "Massachusetts", "MD": "Maryland",
        "ME": "Maine", "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri", "MS": "Mississippi",
        "MT": "Montana", "NC": "North Carolina", "ND": "North Dakota", "NE": "Nebraska",
        "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NV": "Nevada",
        "NY": "New York", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
        "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
        "TX": "Texas", "UT": "Utah", "VA": "Virginia", "VT": "Vermont", "WA": "Washington",
        "WI": "Wisconsin", "WV": "West Virginia", "WY": "Wyoming", "DC": "District of Columbia",
        "AS": "American Samoa", "GU": "Guam GU", "MP": "Northern Mariana Islands",
        "PR": "Puerto Rico", "VI": "U.S. Virgin Islands", "ON": "Canada", "QC": "Canada",
        "BC": "Canada"
    }
    
    def process_df(df):
        df[col] = df[col].astype(str).str.strip()
        df[new_col] = df[col].map(abbreviation_to_name)
        return df

    if isinstance(dfs, list):
        return [process_df(df.copy()) for df in dfs]
    else:
        return process_df(dfs.copy())

def add_regions(dfs, col, new_col):
    """
    Maps region labels based on full State names.
    
    Parameters:
        dfs (pd.DataFrame or list of pd.DataFrame): DataFrame(s) to process.
        col (str): Name of the column with state names (e.g., 'Full State Name').
        new_col (str): Name of the new column to store region.
    
    Returns:
        pd.DataFrame or list of pd.DataFrame: Updated DataFrame(s).
    """  
    state_to_region = {
    'Connecticut': 'Northeast', 'Delaware': 'Northeast', 'District of Columbia': 'Northeast', 'Maine': 'Northeast', 'Maryland': 'Northeast', 
    'Massachusetts': 'Northeast', 'New Hampshire': 'Northeast', 'New Jersey': 'Northeast', 'New York': 'Northeast', 'Pennsylvania': 'Northeast', 
    'Rhode Island': 'Northeast', 'Vermont': 'Northeast',
    'Illinois': 'Central', 'Indiana': 'Central', 'Iowa': 'Central', 'Kansas': 'Central', 'Michigan': 'Central', 'Minnesota': 'Central', 'Missouri': 'Central', 
    'Nebraska': 'Central', 'North Dakota': 'Central', 'Ohio': 'Central', 'South Dakota': 'Central', 'Wisconsin': 'Central',
    'Alabama': 'South', 'Arkansas': 'South', 'Florida': 'South', 'Georgia': 'South', 'Kentucky': 'South', 'Louisiana': 'South', 'Mississippi': 'South', 
    'North Carolina': 'South', 'Oklahoma': 'South', 'Puerto Rico': 'Territories and Possessions', 'South Carolina': 'South', 'Tennessee': 'South', 'Texas': 'South', 
    'Virginia': 'South', 'West Virginia': 'South',
    'Alaska': 'West', 'Arizona': 'West', 'California': 'West', 'Colorado': 'West', 'Hawaii': 'West', 'Idaho': 'West', 'Montana': 'West', 'Nevada': 'West', 
    'New Mexico': 'West', 'Oregon': 'West', 'Utah': 'West', 'Washington': 'West', 'Wyoming': 'West',
    'U.S. Territories and Possessions': 'U.S. Territories and Possessions',
    'Legal Residence is Not in the U.S.': 'Legal Residence is Not in the U.S.',
    'Legal Residence is Unknown': 'Legal Residence is Unknown', 'Canada': 'Legal Residence is Not in the U.S.'
}
    
    def process_df(df):
        df[col] = df[col].astype(str).str.strip()
        df[new_col] = df[col].map(state_to_region)
        return df

    if isinstance(dfs, list):
        return [process_df(df.copy()) for df in dfs]
    else:
        return process_df(dfs.copy())


def A1_pdf(pdf_path, df_name,
           intersection_tolerance=10,
           snap_tolerance=10,
           join_tolerance=35,
           edge_min_length=22,
           min_words_vertical=2,
           min_words_horizontal=1,
           interactive_preview=False):
    """
    Extracts and cleans data from an AAMC A-1 files (pdf format).

    This function:
    - Extracts tables from a multi-page PDF using pdfplumber
    - Cleans and reshapes data, including:
        - Remove headers/footers
        - Remove missing cells and fill in State data
        - Concatenate broken text fields from improper PDF reading
        - Convert numerics with type string to floats/ints
        - Shift data leftward and align columns
        - Drop empty columns/rows
    - Performs additional data manipulation:
        - Add full state names
        - Add regions as defined by the U.S. census to each state
        - Rename columns for better understanding
    - Assigns the cleaned DataFrame to a global variable using the given df_name

    Parameters:
    ----------
    pdf_path (str) : full file path to the AAMC Table A-1 PDF file.

    df_name (str) : variable name for the cleaned DataFrame.

    intersection_tolerance (int) : Controls pixel tolerance for deciding whether vertical and horizontal lines intersect to form table cell corners.  
                                   Increase when: lines are slightly misaligned and cells are not detected properly.  
                                   Decrease when: cells are being merged incorrectly due to loose intersection criteria.
    
    snap_tolerance (int) : Controls how close text or edges must be to a detected line to be considered part of that line.  
                           Increase when: text or edges are slightly off and not snapping correctly to table lines.  
                           Decrease when: snapping is too aggressive and includes unrelated elements.
    
    join_tolerance (int) : Maximum pixel distance between line segments to join broken lines.  
                           Increase when: table borders are broken and need to be connected for proper cell detection.  
                           Decrease when: unrelated lines are being joined, causing errors in table structure.
    
    edge_min_length (int) : Minimum length (pixels) for a line to be considered a valid edge.  
                            Increase when: short noisy lines are detected as edges and causing noise.  
                            Decrease when: small valid edges are being ignored.
    
    min_words_vertical (int) : Minimum number of vertically aligned words to infer a vertical line (column boundary).  
                               Increase when: false columns are detected due to loose criteria.  
                               Decrease when: columns are missed because there are few words vertically aligned.
    
    min_words_horizontal (int) : Minimum number of horizontally aligned words to infer a horizontal line (row boundary).  
                                 Increase when: false rows are detected due to loosely spaced text.  
                                 Decrease when: rows are missed because text is spaced out.
    
    interactive_preview (bool) : If True, shows an initial preview of the extracted raw table and lets the user interactively tweak parameters.

    Returns:
    -------
    df_name (pd.DataFrame) : cleaned DataFrame

    Requirements:
    -------------
    - Needed helper functions:
        `state_abbreviations(df, abbrev_col, full_col)`
        `add_regions(df, state_col, region_col)`
    """
    def extract_tables_with_params():
        pages_list = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                table = page.extract_table({
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "intersection_tolerance": intersection_tolerance,
                    "snap_tolerance": snap_tolerance,
                    "join_tolerance": join_tolerance,
                    "edge_min_length": edge_min_length,
                    "min_words_vertical": min_words_vertical,
                    "min_words_horizontal": min_words_horizontal
                })
                if table:
                    df_page = pd.DataFrame(table[1:])  # skip header row
                    pages_list.append(df_page)
        return pd.concat(pages_list, ignore_index=True).dropna(how='all').dropna(axis=1, how='all')

    # Extract first time for preview or directly if no preview
    df = extract_tables_with_params()

    if interactive_preview:
        pdb.set_trace()
        satisfied = False
        while not satisfied:
            df = extract_tables_with_params()
    
            print("\n--- Extracted table preview ---")
            print(f"\n[Current DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns]")
            print(df.head(10))
    
            user_input = input("\nAre you satisfied with the output? (y to continue, any other key to adjust parameters): ").strip().lower()
            if user_input == 'y':
                satisfied = True
            else:
                def get_int_input(param_name, current_val):
                    inp = input(f"Set '{param_name}' (current {current_val}), or press Enter to keep: ")
                    return int(inp) if inp.strip().isdigit() else current_val
    
                # Re-tweak parameters if needed
                intersection_tolerance = get_int_input("intersection_tolerance", intersection_tolerance)
                snap_tolerance = get_int_input("snap_tolerance", snap_tolerance)
                join_tolerance = get_int_input("join_tolerance", join_tolerance)
                edge_min_length = get_int_input("edge_min_length", edge_min_length)
                min_words_vertical = get_int_input("min_words_vertical", min_words_vertical)
                min_words_horizontal = get_int_input("min_words_horizontal", min_words_horizontal)
    
                print("\nUsing updated parameters:")
                print(f"intersection_tolerance={intersection_tolerance}, snap_tolerance={snap_tolerance}, join_tolerance={join_tolerance}, edge_min_length={edge_min_length}, min_words_vertical={min_words_vertical}, min_words_horizontal={min_words_horizontal}")

    # Convert numeric strings to numbers
    def convert_if_number(x):
        if pd.isna(x): return x
        try:
            if str(x).isdigit(): return int(x)
            return float(x)
        except ValueError:
            return x
    df = df.applymap(convert_if_number)

    def row_has_numeric(row):
        return any(pd.notna(x) and isinstance(x, (int, float, np.integer, np.floating)) for x in row)

    def row_contains_string(row):
        return any('Medical School' in str(x) for x in row if pd.notna(x))

    medical_indices = df.index[df.apply(row_contains_string, axis=1)]
    if len(medical_indices) == 0:
        print("No rows with 'Medical School' found.")
        return None
    df = df.loc[medical_indices[0]:]

    mask_no_numeric = ~df.apply(row_has_numeric, axis=1)
    mask_contains_str = df.apply(row_contains_string, axis=1)
    mask_to_consider = mask_no_numeric & mask_contains_str
    indices = df.index[mask_to_consider]
    if len(indices) > 1:
        df = df.drop(indices[1:])  # Keep only first header row

    def row_is_all_text_or_nan(row):
        return all(pd.isna(x) or isinstance(x, str) for x in row)
    mask_all_text = df.apply(row_is_all_text_or_nan, axis=1)
    all_text_indices = df.index[mask_all_text]
    if len(all_text_indices) > 0:
        keep_mask = (~mask_all_text) | (df.index == all_text_indices[0])
        df = df.loc[keep_mask]

    df = df.iloc[:-1]  # Drop final summary row

    col2, col3 = 2, 3
    def is_number_like(s):
        if not isinstance(s, str): return False
        s = s.strip()
        if not s: return False
        try: float(s.replace(',', '')); return True
        except ValueError: return False

    def concat_col3_to_col2_if_non_numbers(row):
        val2, val3 = row[col2], row[col3]
        if isinstance(val2, str) and isinstance(val3, str):
            if not is_number_like(val2) and not is_number_like(val3):
                return val2 + val3
        return val2

    df['col2_concat'] = df.apply(concat_col3_to_col2_if_non_numbers, axis=1)
    df.rename(columns={'col2_concat': col2}, inplace=True)
    df = df.iloc[:, :-1]

    df[3] = pd.to_numeric(df[3].astype(str).str.replace(',', '', regex=False), errors='coerce')
    df[5] = pd.to_numeric(df[5].astype(str).str.replace(',', '', regex=False), errors='coerce')

    df = df.iloc[1:]

    start_col = 3
    cols_to_shift = df.columns[start_col:]
    for idx, row in df.iterrows():
        values = row[cols_to_shift]
        non_blank = [val for val in values if pd.notna(val) and str(val).strip() != '']
        new_values = non_blank + [np.nan] * (len(cols_to_shift) - len(non_blank))
        df.loc[idx, cols_to_shift] = new_values

    df.dropna(axis=1, how='all', inplace=True)
    df.drop(df.columns[1], axis=1, inplace=True)

    df.columns = ['State', 'Medical School', 'Applications', 'In State Applications (%)',
                  "Out of State Applications (%)", "Male Applicants (%)", "Female Applicants",
                  "Matriculants", "In State Matriculants (%)", "Out of State Matriculants (%)",
                  "Male Matriculants (%)", "Female Matriculants (%)"]

    df = df.apply(pd.to_numeric, errors='ignore')
    df['State'] = df['State'].replace(r'^\s*$', np.nan, regex=True)
    df['State'] = df['State'].ffill()

    df = state_abbreviations(df, 'State', 'State (Full)')
    df = add_regions(df, 'State (Full)', 'Region')

    df.reset_index(drop=True, inplace=True)

    globals()[df_name] = df
    return df

def remove_noNum_rows(df):
    """
    Removes rows from a DataFrame (df) that do not contain at least one cell with ONLY numeric characters

    Parameters:
    -----------
    df (pd.DataFrame) : input DataFrame to filter.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame containing only rows with at least one fully numeric cell.
    """
    def is_only_number(val):
        if pd.isna(val):
            return False
        val_str = str(val).strip()
        return val_str.isdigit()  # True if all characters are digits and not empty

    return df[df.apply(lambda row: any(is_only_number(cell) for cell in row), axis=1)]

def convert_if_number(x):
    if pd.isna(x): return x
    try:
        if str(x).isdigit(): return int(x)
        return float(x)
    except ValueError:
        return x
    
def row_has_numeric(row):
    return any(pd.notna(x) and isinstance(x, (int, float, np.integer, np.floating)) for x in row)

def row_is_all_text_or_nan(row):
    return all(pd.isna(x) or isinstance(x, str) for x in row)

def is_number_like(s):
    if not isinstance(s, str): return False
    s = s.strip()
    if not s: return False
    try:
        float(s.replace(',', ''))
        return True
    except ValueError:
        return False

def concat_col3_to_col2_if_non_numbers(row):
    val2, val3 = row[col2], row[col3]
    if isinstance(val2, str) and isinstance(val3, str):
        if not is_number_like(val2) and not is_number_like(val3):
            return val2 + val3
    return val2

def fix_col_8(row):
    val8 = str(row[8]).strip().rstrip('.')  # remove trailing dot
    val10 = str(row[10]).strip()
    
    digits_to_prepend = ''
    new_val8 = val8
    if '.' in val8:
        parts = val8.split('.')
        if len(parts) == 2 and len(parts[1]) >= 2:
            digits_to_prepend = parts[1][-2:]
            # Remove last 2 digits from decimal part
            new_decimal = parts[1][:-2]
            new_val8 = parts[0] + ('.' + new_decimal if new_decimal else '')
    
    if digits_to_prepend and val10:
        new_val10 = digits_to_prepend + "." + val10
    else:
        new_val10 = val10
    
    return pd.Series([new_val8, new_val10])

school_name_map = {
    "Alabama": "Alabama-Heersink",
    "South Alabama": "South Alabama-Whiddon",
    "Drew": "UCLA Drew",
    "GRU MC Georgia":"Augusta University",
    "MC Georgia Augusta":"Augusta University",
    "Boston":"BU-Chobanian Avedisian",
    "Massachusetts":"Massachusetts-Chan",
    "Mayo":"Mayo-Alix",
    "Nevada":"Nevada Reno",
    "Vermont":"Vermont-Larner",
    "Eastern Virginia":"Eastern Virginia ODU",
    "Utah":"Utah-Eccles",
    "UT Medical Branch":"UT Medical Branch-Sealy",
    "UT Houston":"UT Houston-McGovern",
    "UT HSC San Antonio":"UT San Antonio-Long",
    "Texas A & M":"Texas A&M",
    "South Carolina":"South Carolina Columbia",
    "Temple":"Temple-Katz",
    "Commonwealth":"Geisinger Commonwealth",
    "Yeshiva Einstein":"Einstein",
    "Stony Brook":"Renaissance Stony Brook",
    "SUNY Upstate":"SUNY Upstate-Norton",
    "New York University":"NYU-Grossman",
    "Hofstra North Shore-LIJ":"Zucker Hofstra Northwell",
    "Columbia":"Columbia-Vagelos",
    "Buffalo":"Buffalo-Jacobs",
    "Nevada":"Nevada Reno",
    "Virginia Commonwealt": "Virginia Commonwealth University School of Medicine",
    "Washington State-Floy": "Washington State University Elson S. Floyd College of Medicine",
    "Washington State-Floyd": "Washington State University Elson S. Floyd College of Medicine",
    "Zucker Hofstra Northwell": "Zucker School of Medicine at Hofstra/Northwell",
    "Hofstra Northwell": "Zucker School of Medicine at Hofstra/Northwell",
    "SHU-Hackensack Meridian": "Seton Hall-Hackensack Meridian School of Medicine",
    "Hackensack Meridian": "Hackensack Meridian School of Medicine",
    "Houston-Fertitta": "University of Houston College of Medicine",
    "Houston": "University of Texas Health Science Center at Houston",
    "Kaiser Permanente-Tyson": "Kaiser Permanente School of Medicine",
    "Kaiser Permanente": "Kaiser Permanente School of Medicine",
    "NYU Long Island-Grossman": "NYU Long Island School of Medicine",
    "NYU Long Island": "NYU Long Island School of Medicine",
    "Nevada Las Vegas": "University of Nevada, Las Vegas School of Medicine",
    "TCU UNTHSC": "Texas Christian University and University of North Texas Health Science Center School of Medicine",
    "TCU-Burnett": "Texas Christian University and University of North Texas Health Science Center School of Medicine",
    "UT Medical Branch-Sea": "University of Texas Medical Branch at Galveston",
    "UT Medical Branch-Sealy": "University of Texas Medical Branch at Galveston",
    "UT Austin-Dell": "Dell Medical School at the University of Texas at Austin",
    "UT Rio Grande Valley": "UT Rio Grande Valley School of Medicine",
    "UT Tyler": "University of Texas at Tyler School of Medicine",
    "Belmont-Frist": "Belmont University College of Osteopathic Medicine", 
    "CUNY": "CUNY School of Medicine",
    "California Northstate": "California Northstate University College of Medicine",
    "Carle Illinois": "Carle Illinois College of Medicine",
    "Nova Southeastern-Patel": "Dr. Kiran C. Patel College of Osteopathic Medicine at Nova Southeastern University",
    "Virginia Commonwealth": "Virginia Commonwealth University School of Medicine",
    "University of Texas Medical Branch at Galveston": "UT Medical Branch-Sealy",
    "UT Medical Branch-Galveston": "UT Medical Branch-Sealy",
    "UTMB Galveston": "UT Medical Branch-Sealy",
}

def standardize_school_names(df, mapping):
    df['Medical School'] = df['Medical School'].replace(mapping)
    return df

output_dir = r"C:\Users\yli355\Downloads\analysis\cleaned files"
os.makedirs(output_dir, exist_ok=True)
# %% Load, clean, and export files
# %%% Table 1
table1 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_1.xlsx", 
                              engine=None, 
                              header=4, 
                              na_filter=True, 
                              na_values=[" "])
table1.description = """Shows applicants, matriculants, enrollment, and graduates from 2015-2016 to 2024-2025. 
Enrollment includes the number of students in medical school, including students on a leave of absence, on October 31 of each year. 
Enrollment does not include students with graduated, dismissed, withdrawn, deceased, never enrolled, completed fifth pathway, 
did not complete fifth pathway, or degree revoked statuses."""

table1 = dropNAs(table1).reset_index(drop=True)

table1.to_excel(f"{output_dir}/table1.xlsx", index=False)
# %%% A1

# 2024
A1_2024 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-1.xlsx", )
A1_2024 = remove_noNum_rows(A1_2024)
A1_2024 = A1_2024.iloc[:-1]
    # Rename columns
A1_2024.columns = ['State', 'Medical School', 'Applications', 'In State Applications (%)', 
                  "Out of State Applications (%)", "Male Applicants (%)", "Female Applicants", 
                  "Matriculants", "In State Matriculants (%)", "Out of State Matriculants (%)", 
                  "Male Matriculants (%)", "Female Matriculants (%)"]
    # Forward-fill missing state rows (e.g., when state cell is blank)
A1_2024 = A1_2024.apply(pd.to_numeric, errors='ignore')
A1_2024['State'] = A1_2024['State'].ffill()
    # Add full state names and U.S. census regions
A1_2024 = state_abbreviations(A1_2024, 'State', 'State (Full)')
A1_2024 = add_regions(A1_2024, 'State (Full)', 'Region')
    # Reset index after all cleanup
A1_2024.reset_index(drop=True, inplace=True)

# 2023
A1_2023 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2021_facts_table_a-1.xlsx")
A1_2023 = remove_noNum_rows(A1_2023)
A1_2023 = A1_2023.iloc[:-1]
    # Forward-fill missing state rows (e.g., when state cell is blank)
A1_2023.iloc[:, 0] = A1_2023.iloc[:, 0].ffill()
    # Shift left
A1_2023 = A1_2023.apply(shift_left, axis=1)
A1_2023 = A1_2023.dropna(axis=1, how='all')
    # Rename columns
A1_2023.columns = ['State', 'Medical School', 'Applications', 'In State Applications (%)', 
                  "Out of State Applications (%)", "Male Applicants (%)", "Female Applicants", 
                  "Matriculants", "In State Matriculants (%)", "Out of State Matriculants (%)", 
                  "Male Matriculants (%)", "Female Matriculants (%)"]
    # Add full state names and U.S. census regions
A1_2023 = state_abbreviations(A1_2023, 'State', 'State (Full)')
A1_2023 = add_regions(A1_2023, 'State (Full)', 'Region')
    # Reset index after all cleanup
A1_2023.reset_index(drop=True, inplace=True)

# 2022 
A1_pdf(r"C:\Users\yli355\Downloads\analysis\2022_facts_table_a-1.pdf", "A1_2022", 10,10,35,22,2,1, False)  # sometimes this line will only run correctly when run by itself (highlight and press F9)

# 2021 
    # very sadly the A1_pdf function only worked for that particular pdf so here i converted the pdf to excel in https://www.adobe.com/acrobat/online/pdf-to-excel.html
A1_2021 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2021_facts_table_a-1.xlsx")
A1_2021 = remove_noNum_rows(A1_2021)
A1_2021 = A1_2021.iloc[:-1]
    # Forward-fill missing state rows (e.g., when state cell is blank)
A1_2021.iloc[:, 0] = A1_2021.iloc[:, 0].ffill()
    # shift cells left
A1_2021 = A1_2021.apply(shift_row_left, axis=1)
A1_2021 = A1_2021.dropna(axis=1, how='all')
    # Rename columns
A1_2021.columns = ['State', 'Medical School', 'Applications', 'In State Applications (%)', 
                  "Out of State Applications (%)", "Male Applicants (%)", "Female Applicants", 
                  "Matriculants", "In State Matriculants (%)", "Out of State Matriculants (%)", 
                  "Male Matriculants (%)", "Female Matriculants (%)"]
    # Add full state names and U.S. census regions
A1_2021 = state_abbreviations(A1_2021, 'State', 'State (Full)')
A1_2021 = add_regions(A1_2021, 'State (Full)', 'Region')
    # Reset index after all cleanup
A1_2021.reset_index(drop=True, inplace=True)

# 2020
pages_list = []
with pdfplumber.open(r"C:\Users\yli355\Downloads\analysis\2020_facts_table_a-1.pdf") as pdf:
    for page in pdf.pages:
        table = page.extract_table({
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "intersection_tolerance": 10,
            "snap_tolerance": 10,
            "join_tolerance": 35,
            "edge_min_length": 22,
            "min_words_vertical": 2,
            "min_words_horizontal": 1
        })
        if table:
            df_page = pd.DataFrame(table[1:])  # skip header row
            pages_list.append(df_page)
A1_2020 = pd.concat(pages_list, ignore_index=True).dropna(how='all').dropna(axis=1, how='all')

A1_2020[8] = A1_2020[8].astype(str) + A1_2020[9].astype(str)
A1_2020.drop(columns=9, inplace=True)


A1_2020 = A1_2020.applymap(convert_if_number)

mask_all_text = A1_2020.apply(row_is_all_text_or_nan, axis=1)
all_text_indices = A1_2020.index[mask_all_text]

if len(all_text_indices) > 0:
    keep_mask = (~mask_all_text) | (A1_2020.index == all_text_indices[0])
    A1_2020 = A1_2020.loc[keep_mask]

A1_2020 = A1_2020.drop([174,149,112,75,38,0,1])

col2, col3 = 2, 3

A1_2020[col2] = A1_2020.apply(concat_col3_to_col2_if_non_numbers, axis=1)
A1_2020[3] = pd.to_numeric(A1_2020[3], errors='coerce')

A1_2020.iloc[:, 0].replace(r'^\s*$', pd.NA, regex=True, inplace=True)
A1_2020.iloc[:, 0] = A1_2020.iloc[:, 0].ffill()

# Apply function and assign to both columns at once
A1_2020[[8, 10]] = A1_2020.apply(fix_col_8, axis=1)

A1_2020.at[151, 16] = 30 # fixing some reading errors
A1_2020.at[151, 17] = 100

cols_to_shift = A1_2020.columns[:]
for idx, row in A1_2020.iterrows():
    values = row[cols_to_shift]
    non_blank = [val for val in values if pd.notna(val) and str(val).strip() != '']
    new_values = non_blank + [np.nan] * (len(cols_to_shift) - len(non_blank))
    A1_2020.loc[idx, cols_to_shift] = new_values

A1_2020.dropna(axis=1, how='all', inplace=True)

A1_2020.columns = ['State', 'Medical School', 'Applications', 'In State Applications (%)',
              "Out of State Applications (%)", "Male Applicants (%)", "Female Applicants",
              "Matriculants", "In State Matriculants (%)", "Out of State Matriculants (%)",
              "Male Matriculants (%)", "Female Matriculants (%)"]

A1_2020['Applications'] = A1_2020['Applications'].astype(str).str.replace(',', '', regex=False)
A1_2020 = A1_2020.apply(pd.to_numeric, errors='ignore')

A1_2020 = state_abbreviations(A1_2020, 'State', 'State (Full)')
A1_2020 = add_regions(A1_2020, 'State (Full)', 'Region')

A1_2020.reset_index(drop=True, inplace=True)
    
# 2019
A1_2019 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2019_facts_table_a-1.xlsx")
A1_2019 = remove_noNum_rows(A1_2019)
A1_2019 = A1_2019.iloc[1:-1]
A1_2019 = A1_2019.drop([49,96,143,190])
    # Forward-fill missing state rows (e.g., when state cell is blank)
A1_2019.iloc[:, 0] = A1_2019.iloc[:, 0].ffill()
    # Rename columns
A1_2019.columns = ['State', 'Medical School', 'Applications', 'In State Applications (%)', 
                  "Out of State Applications (%)", "Male Applicants (%)", "Female Applicants", 
                  "Matriculants", "In State Matriculants (%)", "Out of State Matriculants (%)", 
                  "Male Matriculants (%)", "Female Matriculants (%)"]
    # Add full state names and U.S. census regions
A1_2019 = state_abbreviations(A1_2019, 'State', 'State (Full)')
A1_2019 = add_regions(A1_2019, 'State (Full)', 'Region')
    # Reset index after all cleanup
A1_2019.reset_index(drop=True, inplace=True)
    # Make numberic
A1_2019 = A1_2019.apply(pd.to_numeric, errors='ignore')

# 2018
pages_list = []
A1_2018 = pd.DataFrame()
with pdfplumber.open(r"C:\Users\yli355\Downloads\analysis\2020_facts_table_a-1.pdf") as pdf:
    for page in pdf.pages:
        table = page.extract_table({
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "intersection_tolerance": 10,
            "snap_tolerance": 10,
            "join_tolerance": 40,
            "edge_min_length": 20,
            "min_words_vertical": 2,
            "min_words_horizontal": 1
        })
        if table:
            df_page = pd.DataFrame(table[1:])  # skip header row
            pages_list.append(df_page)
A1_2018 = pd.concat(pages_list, ignore_index=True).dropna(how='all').dropna(axis=1, how='all')

A1_2018[8] = A1_2018[8].astype(str) + A1_2018[9].astype(str)
A1_2018.drop(columns=9, inplace=True)

    # Convert numeric strings to numbers
A1_2018 = A1_2018.applymap(convert_if_number)

mask_all_text = A1_2018.apply(row_is_all_text_or_nan, axis=1)
all_text_indices = A1_2018.index[mask_all_text]

if len(all_text_indices) > 0:
    keep_mask = (~mask_all_text) | (A1_2018.index == all_text_indices[0])
    A1_2018 = A1_2018.loc[keep_mask]

A1_2018 = A1_2018.drop([174,149,112,75,38,0,1])

col2, col3 = 2, 3

A1_2018[col2] = A1_2018.apply(concat_col3_to_col2_if_non_numbers, axis=1)
A1_2018.iloc[:, 3] = pd.to_numeric(A1_2018.iloc[:, 3], errors='coerce')

A1_2018.iloc[:, 0].replace(r'^\s*$', pd.NA, regex=True, inplace=True)
A1_2018.iloc[:, 0] = A1_2018.iloc[:, 0].ffill()

    # Apply function and assign to both columns at once
A1_2018[[8, 10]] = A1_2018.apply(fix_col_8, axis=1)

A1_2018.at[151, 16] = 30 # fixing some reading errors
A1_2018.at[151, 17] = 100

cols_to_shift = A1_2018.columns[:]
for idx, row in A1_2018.iterrows():
    values = row[cols_to_shift]
    non_blank = [val for val in values if pd.notna(val) and str(val).strip() != '']
    new_values = non_blank + [np.nan] * (len(cols_to_shift) - len(non_blank))
    A1_2018.loc[idx, cols_to_shift] = new_values

A1_2018.dropna(axis=1, how='all', inplace=True)

A1_2018.columns = ['State', 'Medical School', 'Applications', 'In State Applications (%)',
              "Out of State Applications (%)", "Male Applicants (%)", "Female Applicants",
              "Matriculants", "In State Matriculants (%)", "Out of State Matriculants (%)",
              "Male Matriculants (%)", "Female Matriculants (%)"]

A1_2018['Applications'] = A1_2018['Applications'].astype(str).str.replace(',', '', regex=False)
A1_2018 = A1_2018.apply(pd.to_numeric, errors='ignore')

A1_2018 = state_abbreviations(A1_2018, 'State', 'State (Full)')
A1_2018 = add_regions(A1_2018, 'State (Full)', 'Region')

A1_2018.reset_index(drop=True, inplace=True)

# 2017
pages_list = []
A1_2017 = pd.DataFrame()
with pdfplumber.open(r"C:\Users\yli355\Downloads\analysis\2017_facts_table_a-1.pdf") as pdf:
    for page in pdf.pages:
        table = page.extract_table({
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "intersection_tolerance": 8,
            "snap_tolerance": 8,
            "join_tolerance": 10,
            "edge_min_length": 10,
            "min_words_vertical": 2,
            "min_words_horizontal": 1
        })
        if table:
            df_page = pd.DataFrame(table[1:])  # skip header row
            pages_list.append(df_page)
A1_2017 = pd.concat(pages_list, ignore_index=True).dropna(how='all').dropna(axis=1, how='all')

A1_2017[14] = A1_2017[14].astype(str) + A1_2017[15].astype(str)
A1_2017.drop(columns=15, inplace=True)

A1_2017 = A1_2017.applymap(convert_if_number)

mask_all_text = A1_2017.apply(row_is_all_text_or_nan, axis=1)
all_text_indices = A1_2017.index[mask_all_text]

if len(all_text_indices) > 0:
    keep_mask = (~mask_all_text) | (A1_2017.index == all_text_indices[0])
    A1_2017 = A1_2017.loc[keep_mask]
    
A1_2017 = A1_2017.drop([0,1,3,38,41,76,79,114,117,152,155,176])

A1_2017.iloc[:, 0].replace(r'^\s*$', pd.NA, regex=True, inplace=True)
A1_2017.iloc[:, 0] = A1_2017.iloc[:, 0].ffill()

cols_to_shift = A1_2017.columns[:]
for idx, row in A1_2017.iterrows():
    values = row[cols_to_shift]
    non_blank = [val for val in values if pd.notna(val) and str(val).strip() != '']
    new_values = non_blank + [np.nan] * (len(cols_to_shift) - len(non_blank))
    A1_2017.loc[idx, cols_to_shift] = new_values

A1_2017.dropna(axis=1, how='all', inplace=True)

A1_2017.columns = ['State', 'Medical School', 'Applications', 'In State Applications (%)',
              "Out of State Applications (%)", "Male Applicants (%)", "Female Applicants",
              "Matriculants", "In State Matriculants (%)", "Out of State Matriculants (%)",
              "Male Matriculants (%)", "Female Matriculants (%)"]

A1_2017['Applications'] = A1_2017['Applications'].astype(str).str.replace(',', '', regex=False)
A1_2017 = A1_2017.apply(pd.to_numeric, errors='ignore')

A1_2017 = state_abbreviations(A1_2017, 'State', 'State (Full)')
A1_2017 = add_regions(A1_2017, 'State (Full)', 'Region')

A1_2017.reset_index(drop=True, inplace=True)

# 2016
pages_list = []
A1_2016 = pd.DataFrame()
with pdfplumber.open(r"C:\Users\yli355\Downloads\analysis\2016_facts_table_a-1.pdf") as pdf:
    for page in pdf.pages:
        table = page.extract_table({
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "intersection_tolerance": 8,
            "snap_tolerance": 8,
            "join_tolerance": 10,
            "edge_min_length": 10,
            "min_words_vertical": 2,
            "min_words_horizontal": 2
        })
        if table:
            df_page = pd.DataFrame(table[1:])  # skip header row
            pages_list.append(df_page)
A1_2016 = pd.concat(pages_list, ignore_index=True).dropna(how='all').dropna(axis=1, how='all')

A1_2016[7] = A1_2016[7].astype(str) + A1_2016[8].astype(str)
A1_2016[7] = A1_2016[7].astype(str) + A1_2016[9].astype(str)
A1_2016.drop(columns=8, inplace=True)
A1_2016.drop(columns=9, inplace=True)

A1_2016.at[181, 1] = 'Virginia Commonwealth'
A1_2016.at[181, 2] = ''

A1_2016 = A1_2016.applymap(convert_if_number)

mask_all_text = A1_2016.apply(row_is_all_text_or_nan, axis=1)
all_text_indices = A1_2016.index[mask_all_text]

if len(all_text_indices) > 0:
    keep_mask = (~mask_all_text) | (A1_2016.index == all_text_indices[0])
    A1_2016 = A1_2016.loc[keep_mask]

A1_2016 = A1_2016.drop([0,1,4,42,46,84,88,126,130,168,172,189])

A1_2016.iloc[:, 0].replace(r'^\s*$', pd.NA, regex=True, inplace=True)
A1_2016.iloc[:, 0] = A1_2016.iloc[:, 0].ffill()

cols_to_shift = A1_2016.columns[:]
for idx, row in A1_2016.iterrows():
    values = row[cols_to_shift]
    non_blank = [val for val in values if pd.notna(val) and str(val).strip() != '']
    new_values = non_blank + [np.nan] * (len(cols_to_shift) - len(non_blank))
    A1_2016.loc[idx, cols_to_shift] = new_values
    
A1_2016.dropna(axis=1, how='all', inplace=True)
    
A1_2016.columns = ['State', 'Medical School', 'Applications', 'In State Applications (%)',
              "Out of State Applications (%)", "Male Applicants (%)", "Female Applicants",
              "Matriculants", "In State Matriculants (%)", "Out of State Matriculants (%)",
              "Male Matriculants (%)", "Female Matriculants (%)"]

A1_2016['Applications'] = A1_2016['Applications'].astype(str).str.replace(',', '', regex=False)
A1_2016 = A1_2016.apply(pd.to_numeric, errors='ignore')

A1_2016 = state_abbreviations(A1_2016, 'State', 'State (Full)')
A1_2016 = add_regions(A1_2016, 'State (Full)', 'Region')

A1_2016.reset_index(drop=True, inplace=True)

# 2015
pages_list = []
A1_2015 = pd.DataFrame()
with pdfplumber.open(r"C:\Users\yli355\Downloads\analysis\2015_facts_table_a-1.pdf") as pdf:
    for page in pdf.pages:
        table = page.extract_table({
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "intersection_tolerance": 8,
            "snap_tolerance": 8,
            "join_tolerance": 20,
            "edge_min_length": 10,
            "min_words_vertical": 2,
            "min_words_horizontal": 2
        })
        if table:
            df_page = pd.DataFrame(table[1:])  # skip header row
            pages_list.append(df_page)
A1_2015 = pd.concat(pages_list, ignore_index=True).dropna(how='all').dropna(axis=1, how='all')

A1_2015.at[164, 1] = 'Virginia Commonwealth'
A1_2015.at[164, 2] = ''

A1_2015[2] = A1_2015[2].astype(str) + A1_2015[3].astype(str)
A1_2015[4] = A1_2015[4].astype(str) + A1_2015[5].astype(str)
A1_2015.drop(columns=3, inplace=True)
A1_2015.drop(columns=5, inplace=True)

A1_2015 = A1_2015.applymap(convert_if_number)

mask_all_text = A1_2015.apply(row_is_all_text_or_nan, axis=1)
all_text_indices = A1_2015.index[mask_all_text]

if len(all_text_indices) > 0:
    keep_mask = (~mask_all_text) | (A1_2015.index == all_text_indices[0])
    A1_2015 = A1_2015.loc[keep_mask]

A1_2015 = A1_2015.drop([0,2,40,78,116,154,172])

A1_2015.at[162,0] = 'VA'
A1_2015.at[162,1] = 'Eastern Virginia'

A1_2015.iloc[:, 0].replace(r'^\s*$', pd.NA, regex=True, inplace=True)
A1_2015.iloc[:, 0] = A1_2015.iloc[:, 0].ffill()

A1_2015[19] = A1_2015[19].astype(str) + A1_2015[20].astype(str)
A1_2015[21] = A1_2015[21].astype(str) + A1_2015[22].astype(str)
A1_2015.drop(columns=20, inplace=True)
A1_2015.drop(columns=22, inplace=True)

A1_2015['second_number'] = A1_2015[14].str.split().str[1]
A1_2015['old14'] = A1_2015[14].str.split().str[0]

A1_2015[14]=A1_2015['old14']
A1_2015.drop('old14', axis=1, inplace=True) 

A1_2015[15] = A1_2015['second_number'].astype(str) + A1_2015[15].astype(str)
A1_2015.drop('second_number', axis=1, inplace=True)

cols_to_shift = A1_2015.columns[:]
for idx, row in A1_2015.iterrows():
    values = row[cols_to_shift]
    non_blank = [val for val in values if pd.notna(val) and str(val).strip() != '']
    new_values = non_blank + [np.nan] * (len(cols_to_shift) - len(non_blank))
    A1_2015.loc[idx, cols_to_shift] = new_values
    
A1_2015.dropna(axis=1, how='all', inplace=True)
A1_2015.drop(15, axis=1, inplace=True) 
A1_2015 = A1_2015.replace('nan', np.nan)
    
cols_to_shift = A1_2015.columns[:]
for idx, row in A1_2015.iterrows():
    values = row[cols_to_shift]
    non_blank = [val for val in values if pd.notna(val) and str(val).strip() != '']
    new_values = non_blank + [np.nan] * (len(cols_to_shift) - len(non_blank))
    A1_2015.loc[idx, cols_to_shift] = new_values
A1_2015.dropna(axis=1, how='all', inplace=True)

A1_2015.columns = ['State', 'Medical School', 'Applications', 'In State Applications (%)',
              "Out of State Applications (%)", "Male Applicants (%)", "Female Applicants",
              "Matriculants", "In State Matriculants (%)", "Out of State Matriculants (%)",
              "Male Matriculants (%)", "Female Matriculants (%)"]

A1_2015['Applications'] = A1_2015['Applications'].astype(str).str.replace(',', '', regex=False)
A1_2015 = A1_2015.apply(pd.to_numeric, errors='ignore')

A1_2015 = state_abbreviations(A1_2015, 'State', 'State (Full)')
A1_2015 = add_regions(A1_2015, 'State (Full)', 'Region')

A1_2015.reset_index(drop=True, inplace=True)

# 2014
pages_list = []
A1_2014 = pd.DataFrame()
with pdfplumber.open(r"C:\Users\yli355\Downloads\analysis\2014_facts_table_1.pdf") as pdf:
    for page in pdf.pages:
        table = page.extract_table({
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "intersection_tolerance": 8,
            "snap_tolerance": 8,
            "join_tolerance": 32,
            "edge_min_length": 10,
            "min_words_vertical": 4,
            "min_words_horizontal": 7
        })
        if table:
            df_page = pd.DataFrame(table[1:])  # skip header row
            pages_list.append(df_page)
A1_2014 = pd.concat(pages_list, ignore_index=True).dropna(how='all').dropna(axis=1, how='all')

A1_2014 = A1_2014.applymap(convert_if_number)

mask_all_text = A1_2014.apply(row_is_all_text_or_nan, axis=1)
all_text_indices = A1_2014.index[mask_all_text]

if len(all_text_indices) > 0:
    keep_mask = (~mask_all_text) | (A1_2014.index == all_text_indices[0])
    A1_2014 = A1_2014.loc[keep_mask]
    
A1_2014 = A1_2014.iloc[1:-1]
A1_2014.drop(6, axis=1, inplace=True)

A1_2014.iloc[:,0].replace(r'^\s*$',pd.NA, regex=True, inplace=True)
A1_2014.iloc[:,0] = A1_2014.iloc[:,0].ffill()

A1_2014.at[54,8] = 72
A1_2014.at[54,9] = 10

A1_2014.columns = ['State', 'Medical School', 'Applications', 'In State Applications (%)',
              "Out of State Applications (%)", "Male Applicants (%)", "Female Applicants",
              "Matriculants", "In State Matriculants (%)", "Out of State Matriculants (%)",
              "Male Matriculants (%)", "Female Matriculants (%)"]

A1_2014['Applications'] = A1_2014['Applications'].astype(str).str.replace(',', '', regex=False)
A1_2014 = A1_2014.apply(pd.to_numeric, errors='ignore')

A1_2014 = state_abbreviations(A1_2014, 'State', 'State (Full)')
A1_2014 = add_regions(A1_2014, 'State (Full)', 'Region')

A1_2014.reset_index(drop=True, inplace=True)

A1_2014.at[12,'Medical School'] = 'UC San Francisco'
A1_2014.at[14,'Medical School'] = 'UCLA-Geffen'
A1_2014.at[14,'Medical School'] = 'UCLA-Missouri Kansas City'
A1_2014.at[65,'Medical School'] = 'Missouri Kansas City'

years = list(range(2014, 2025))

for year in years:
    df_name = f"A1_{year}"
    globals()[df_name] = standardize_school_names(globals()[df_name], school_name_map)

years = list(range(2014, 2025))
dfs = [globals()[f"A1_{year}"].copy() for year in years]

for year, df in zip(years, dfs):
    df["Year"] = year

school_names_by_year = {year: set(df['Medical School'].unique()) for year, df in zip(years, dfs)}
all_schools = sorted(set.union(*school_names_by_year.values()))

print("=== Missing Schools by Year ===")

for year in years:
    missing = set(all_schools) - school_names_by_year[year]
    if missing:
        print(f"{year}: Missing {len(missing)} schools → {sorted(missing)}")
    else:
        print(f"{year}: All schools present")

with pd.ExcelWriter(os.path.join(output_dir, "medical_school_by_year.xlsx"), engine="xlsxwriter") as writer:
    for year, df in zip(years, dfs):
        df.to_excel(writer, sheet_name=str(year), index=False)

all_data = pd.concat(dfs, ignore_index=True)
grouped = all_data.groupby("Medical School")

with pd.ExcelWriter(os.path.join(output_dir, "medical_school_by_school.xlsx"), engine="xlsxwriter") as writer:
    for school, group in grouped:
        sheet_name = school[:31].replace("/", "-").replace("\\", "-")  # Clean sheet names
        group.sort_values("Year").to_excel(writer, sheet_name=sheet_name, index=False)

# %%% A2

A2 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-2.xlsx", header=4)
A2 = sepxyz(A2,'Undergraduate Institution', 'University', 'City', 'State')
A2 = A2.iloc[:-1]

A2o1 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-2.1.xlsx", header=4)
A2o1 = sepxyz(A2o1,'Undergraduate Institution', 'University', 'City', 'State')
A2o1 = A2o1.iloc[:-1]
A2o1 = A2o1.rename(columns={
    'Black or African American\nApplicants from the Institution': 'Black or African American Applicants from the Institution',
    'Total Applicants from the \nInstitution':'Total Applicants from the Institution'
    })

A2o2 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-2.2.xlsx", header=4)
A2o2 = sepxyz(A2o2,'Undergraduate Institution', 'University', 'City', 'State')
A2o2 = A2o2.iloc[:-1]
A2o1 = A2o1.rename(columns={
    'American Indian or Alaska Native Applicants from the\n Institution': 'American Indian or Alaska Native Applicants from the Institution',
    'Total Applicants from the \nInstitution':'Total Applicants from the Institution'
    })


A2o3 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-2.3.xlsx", header=4)
A2o3 = sepxyz(A2o3,'Undergraduate Institution', 'University', 'City', 'State')
A2o3 = A2o3.iloc[:-1]
A2o3 = A2o3.rename(columns={
    '% of All Hispanic, Latino, or \nof Spanish Origin Applicants\n to U.S. MD-Granting Medical Schools': '% of All Hispanic, Latino, or Spanish Origin Applicants to U.S. MD-Granting Medical Schools',
    'Total Applicants from the \nInstitution':'Total Applicants from the Institution'
    })

A2o4 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-2.4.xlsx", header=4)
A2o4 = sepxyz(A2o4,'Undergraduate Institution', 'University', 'City', 'State')
A2o4 = A2o4.iloc[:-1]
A2o4 = A2o4.rename(columns={
    '% of All Asian \nApplicants to U.S. MD-Granting Medical Schools': '% of All Asian Applicants to U.S. MD-Granting Medical Schools',
    'Asian\nApplicants from the Institution': 'Asian Applicants from the Institution',
    'Total Applicants from the \nInstitution':'Total Applicants from the Institution'
    })

A2o5 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-2.5.xlsx", header=4)
A2o5 = sepxyz(A2o5,'Undergraduate Institution', 'University', 'City', 'State')
A2o5 = A2o5.iloc[:-1]
A2o5 = A2o5.rename(columns={
    'White\nApplicants from the Institution': 'White Applicants from the Institution',
    '% of All White\nApplicants to U.S. MD-Granting Medical Schools': '% of All White Applicants to U.S. MD-Granting Medical Schools',
    'Total Applicants from the \nInstitution':'Total Applicants from the Institution'
    })

replacements = {
    'University': {'University at Albany': 'University at Albany, SUNY'},
    'City': {' SUNY': 'Albany'},
    'State': {' Albany, NY': 'NY'}
}

for df in [A2, A2o1, A2o2, A2o3, A2o4, A2o5]:
    for col, repl in replacements.items():
            df[col] = df[col].replace(repl)

A2.rename(columns={'Total Applicants from the \nInstitution': 'Total Applicants from the Institution'}, inplace=True)

# Add column for full state name
A2, A2o1, A2o2, A2o3, A2o4, A2o5 = state_abbreviations([A2, A2o1, A2o2, A2o3, A2o4, A2o5], 'State', 'Full State Name') 

# Add regional labels to states
A2, A2o1, A2o2, A2o3, A2o4, A2o5 = add_regions([A2, A2o1, A2o2, A2o3, A2o4, A2o5], 'Full State Name', 'Region (School)') 

# combine files
temp = pd.DataFrame()

temp = A2.merge(A2o1, how='outer', on='University', suffixes=('', '_A2.1'))
temp['City'] = temp['City'].combine_first(temp['City_A2.1'])
temp['State'] = temp['State'].combine_first(temp['State_A2.1'])
temp['Region (School)'] = temp['Region (School)'].combine_first(temp['Region (School)_A2.1'])
temp['Full State Name'] = temp['Full State Name'].combine_first(temp['Full State Name_A2.1'])
temp['Total Applicants from the Institution'] = temp['Total Applicants from the Institution'].combine_first(temp['Total Applicants from the Institution_A2.1'])
temp.drop(['Total Applicants from the Institution_A2.1', 'City_A2.1', 'State_A2.1', 'Region (School)_A2.1','Full State Name_A2.1'], axis=1, inplace=True)

temp = temp.merge(A2o2, how='outer', on='University', suffixes=('', '_A2.2'))
temp['City'] = temp['City'].combine_first(temp['City_A2.2'])
temp['State'] = temp['State'].combine_first(temp['State_A2.2'])
temp['Region (School)'] = temp['Region (School)'].combine_first(temp['Region (School)_A2.2'])
temp['Full State Name'] = temp['Full State Name'].combine_first(temp['Full State Name_A2.2'])
temp['Total Applicants from the Institution'] = temp['Total Applicants from the Institution'].combine_first(temp['Total Applicants from the \nInstitution'])
temp.drop(['Total Applicants from the \nInstitution', 'City_A2.2', 'State_A2.2', 'Region (School)_A2.2','Full State Name_A2.2'], axis=1, inplace=True)
temp.rename(columns={'American Indian or Alaska Native Applicants from the\n Institution': 'American Indian or Alaska Native Applicants from the Institutionn'}, inplace=True)

temp = temp.merge(A2o3, how='outer', on='University', suffixes=('', '_A2.3'))
temp['City'] = temp['City'].combine_first(temp['City_A2.3'])
temp['State'] = temp['State'].combine_first(temp['State_A2.3'])
temp['Region (School)'] = temp['Region (School)'].combine_first(temp['Region (School)_A2.3'])
temp['Full State Name'] = temp['Full State Name'].combine_first(temp['Full State Name_A2.3'])
temp['Total Applicants from the Institution'] = temp['Total Applicants from the Institution'].combine_first(temp['Total Applicants from the Institution_A2.3'])
temp.drop(['Total Applicants from the Institution_A2.3', 'City_A2.3', 'State_A2.3', 'Region (School)_A2.3','Full State Name_A2.3'], axis=1, inplace=True)

temp = temp.merge(A2o3, how='outer', on='University', suffixes=('', '_A2.4'))
temp['City'] = temp['City'].combine_first(temp['City_A2.4'])
temp['State'] = temp['State'].combine_first(temp['State_A2.4'])
temp['Region (School)'] = temp['Region (School)'].combine_first(temp['Region (School)_A2.4'])
temp['Full State Name'] = temp['Full State Name'].combine_first(temp['Full State Name_A2.4'])
temp['Total Applicants from the Institution'] = temp['Total Applicants from the Institution'].combine_first(temp['Total Applicants from the Institution_A2.4'])
temp.drop(['Total Applicants from the Institution_A2.4', 'City_A2.4', 'State_A2.4', 'Region (School)_A2.4','Full State Name_A2.4', 'Hispanic, Latino, or of Spanish Origin Applicants from the Institution_A2.4', '% of All Hispanic, Latino, or Spanish Origin Applicants to U.S. MD-Granting Medical Schools_A2.4'], axis=1, inplace=True)

temp = temp.merge(A2o4, how='outer', on='University', suffixes=('', '_A2.4'))
temp['City'] = temp['City'].combine_first(temp['City_A2.4'])
temp['State'] = temp['State'].combine_first(temp['State_A2.4'])
temp['Region (School)'] = temp['Region (School)'].combine_first(temp['Region (School)_A2.4'])
temp['Full State Name'] = temp['Full State Name'].combine_first(temp['Full State Name_A2.4'])
temp['Total Applicants from the Institution'] = temp['Total Applicants from the Institution'].combine_first(temp['Total Applicants from the Institution_A2.4'])
temp.drop(['Total Applicants from the Institution_A2.4', 'City_A2.4', 'State_A2.4', 'Region (School)_A2.4','Full State Name_A2.4'], axis=1, inplace=True)

temp = temp.merge(A2o5, how='outer', on='University', suffixes=('', '_A2.5'))
temp['City'] = temp['City'].combine_first(temp['City_A2.5'])
temp['State'] = temp['State'].combine_first(temp['State_A2.5'])
temp['Region (School)'] = temp['Region (School)'].combine_first(temp['Region (School)_A2.5'])
temp['Full State Name'] = temp['Full State Name'].combine_first(temp['Full State Name_A2.5'])
temp['Total Applicants from the Institution'] = temp['Total Applicants from the Institution'].combine_first(temp['Total Applicants from the Institution_A2.5'])
temp.drop(['Total Applicants from the Institution_A2.5', 'City_A2.5', 'State_A2.5', 'Region (School)_A2.5','Full State Name_A2.5'], axis=1, inplace=True)

Final2 = temp.copy()
temp = pd.DataFrame()

# Add % of total applicants to A2
Total_Applicants_25 = table1.at[0, '2024-2025']
Final2['% of All Applicants to U.S. MD-Granting Medical Schools'] = Final2['Total Applicants from the Institution']*100/Total_Applicants_25

output_path = os.path.join(output_dir, "Applicant_Race_Info.xlsx")
Final2.to_excel(output_path, index=False)

# A3 to A4
A3 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-3.xlsx", header=0)
A3.description = """Applicants to U.S. Medical Schools by State of Legal Residence, 2015-2016 through 2024-2025"""
A3.drop(A3.columns[-1], axis=1, inplace=True)

A4 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-4.xlsx", header=0)
A4.description = """Matriculants to U.S. Medical Schools by State of Legal Residence, 2015-2016 through 2024-2025"""
A4.drop(A4.columns[-1], axis=1, inplace=True)

A3_4 = A4.iloc[:,:2].copy()
percent_matriculated = A4.iloc[:,2:] * 100 / A3.iloc[:,2:]
A3_4 = pd.concat([A3_4, percent_matriculated], axis=1)

output_path = os.path.join(output_dir, "Matriculation and Application info by state 2015 to 2025.xlsx")

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    A3.to_excel(writer, sheet_name="Applicants", index=False)
    A4.to_excel(writer, sheet_name="Matriculants", index=False)
    A3_4.to_excel(writer, sheet_name="% Matriculated", index=False)
    
# A5
A5_2024 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-5.xlsx", header=0)
A5_2024.description = """The table below displays the numbers of applicants in 2025-2026 by state of legal residence and their in-state or out-of-state matriculation status."""

A5_2023 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2023_FACTS_Table_A-5.xlsx", header=0)

pages_list = []
A5_2018 = pd.DataFrame()
with pdfplumber.open(r"C:\Users\yli355\Downloads\analysis\2018_facts_table_a-5.pdf") as pdf:
    for page in pdf.pages:
        table = page.extract_table({
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "intersection_tolerance": 8,
            "snap_tolerance": 8,
            "join_tolerance": 25,
            "edge_min_length": 3,
            "min_words_vertical": 4,
            "min_words_horizontal": 5
        })
        if table:
            df_page = pd.DataFrame(table[1:])  # skip header row
            pages_list.append(df_page)
A5_2018 = pd.concat(pages_list, ignore_index=True).dropna(how='all').dropna(axis=1, how='all')
rows_to_drop = (list(range(1, 3)) +[15, 16] + list(range(26, 32)) +list(range(36, 37)) +list(range(53, 60)) +list(range(74, 76)) +list(range(79, 85)))
A5_2018 = A5_2018.drop(index=rows_to_drop)
A5_2018 = A5_2018.drop([0,32,37,60,85])
A5_2018 = A5_2018.apply(shift_right_if_next_is_nan, axis=1)
A5_2018 = A5_2018.apply(shift_right_if_next_is_nan, axis=1)
A5_2018_a = A5_2023.iloc[:,:2].copy()
A5_2018_a.reset_index(drop=True, inplace=True)
A5_2018_b = A5_2018.iloc[:,3:].copy()
A5_2018_b.reset_index(drop=True, inplace=True)
A5_2018_b = A5_2018_b.dropna(axis=0, how='all')
A5_2018 = pd.DataFrame()
A5_2018 = pd.concat([A5_2018_a, A5_2018_b], axis = 1)
A5_2018.columns = ['Region', 'State of Legal Residence', 'Applicants',
       'Matriculated In-State (n)', 'Matriculated In-State (%)',
       'Matriculated Out-of-State (n)', 'Matriculated Out-of-State (%)',
       'Did Not Matriculate to Any U.S. MD-Granting Medical School (n)',
       'Did Not Matriculate to Any U.S. MD-Granting Medical School (%)']

pages_list = []
A5_2014 = pd.DataFrame()
with pdfplumber.open(r"C:\Users\yli355\Downloads\analysis\2014_facts_table_5.pdf") as pdf:
    for page in pdf.pages:
        table = page.extract_table({
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "intersection_tolerance": 8,
            "snap_tolerance": 8,
            "join_tolerance": 25,
            "edge_min_length": 3,
            "min_words_vertical": 4,
            "min_words_horizontal": 5
        })
        if table:
            df_page = pd.DataFrame(table[1:])  # skip header row
            pages_list.append(df_page)
A5_2014 = pd.concat(pages_list, ignore_index=True).dropna(how='all').dropna(axis=1, how='all')
A5_2014 = A5_2014.drop([0,1,2,15,16,29,30,46,47,48,49,63,64,68])
A5_2014.iloc[:,0].replace(r'^\s*$',pd.NA, regex=True, inplace=True)
A5_2014.iloc[:,0] = A5_2014.iloc[:,0].ffill()
A5_2014.at[65,0] = "U.S. Territories and Possessions"
A5_2014.at[65,1] = "U.S. Territories and Possessions"
A5_2014.at[66,0] = "Legal Residence is Not in the U.S."
A5_2014.at[66,1] = "Legal Residence is Not in the U.S."
A5_2014.at[67,0] = "Legal Residence is Unknown"
A5_2014.at[67,1] = "Legal Residence is Unknown"

output_path = os.path.join(output_dir, "In vs out of state matriculation.xlsx")

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    A5_2024.to_excel(writer, sheet_name="2024", index=False)
    A5_2023.to_excel(writer, sheet_name="2023", index=False)
    A5_2018.to_excel(writer, sheet_name="2018", index=False)
    A5_2014.to_excel(writer, sheet_name="2014", index=False)

# A7
A7o1 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-7.1.xlsx", header=0)
A7o1.description = """Applicants, First-Time Applicants, Acceptees, and Matriculants to U.S. Medical Schools by Gender, 2005-2006 through 2014-2015"""
new_rows = [
    {'Type': "Applicants", 'Gender': "Another Gender Identity"},
    {'Type': "Acceptees", 'Gender': "Another Gender Identity"},
    {'Type': "First-Time Applicants          ", 'Gender': "Another Gender Identity"},
    {'Type': "Matriculants", 'Gender': "Another Gender Identity"}
]
A7o1 = pd.concat([A7o1, pd.DataFrame(new_rows)], ignore_index=True)
A7o1 = A7o1.sort_values(by='Type').reset_index(drop=True)

A7o2 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-7.2.xlsx", header=0)
A7o2.description = """Applicants, First-Time Applicants, Acceptees, and Matriculants to U.S. Medical Schools by Gender, 2015-2016 through 2024-2025"""
A7o2 = A7o2.sort_values(by='Applicants, First-Time Applicants, Acceptees, and Matriculants').reset_index(drop=True)
A7o2 = A7o2.iloc[:,2:]

A7 = pd.concat([A7o1, A7o2], axis=1)
A7.replace('-','', regex=True, inplace=True)
A7 = A7.replace(np.nan, '') 

output_path = os.path.join(output_dir, "Gender_over_Time.xlsx")
A7.to_excel(output_path, index=False)

# A12
A12 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-12.xlsx", header=[0, 1])
A12.description = """Applicants, First-Time Applicants, Acceptees, and Matriculants to U.S. Medical Schools by Race/Ethnicity, 2021-2022 through 2024-2025"""
A12.drop(A12.columns[2], axis=1, inplace=True)
A12 = A12.groupby([A12.columns[0], A12.columns[1]], as_index=False).sum()
year_intervals = A12.columns[2:].levels[0]  # Skip first two regular columns
yearly_dfs = {}

for interval in year_intervals:
    year_cols = A12.loc[:, (interval,)]
    A12_df = pd.concat([A12.iloc[:, :2], year_cols], axis=1)
    start_year = interval.split('-')[0]
    yearly_dfs[f"A12_{start_year}"] = A12_df

A12_2021 = yearly_dfs["A12_2021"]
A12_2022 = yearly_dfs["A12_2022"]
A12_2023 = yearly_dfs["A12_2023"]
A12_2024 = yearly_dfs["A12_2024"]

output_path = os.path.join(output_dir, "Race_ethnicity_info.xlsx")

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    A12.to_excel(writer, sheet_name="Combined")
    A12_2021.to_excel(writer, sheet_name="2021-2022")
    A12_2022.to_excel(writer, sheet_name="2022-2023")
    A12_2023.to_excel(writer, sheet_name="2023-2024")
    A12_2024.to_excel(writer, sheet_name="2024-2025")
    
# A16
A16 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-16.xlsx",header=0)
A16.description = """MCAT Scores and GPAs for Applicants and Matriculants to U.S. Medical Schools, 2020-2021 through 2024-2025"""
output_path = os.path.join(output_dir, "Overall MCAT Scores and GPAs.xlsx")
A16.to_excel(output_path)

# A17
A17 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-17.xlsx", header=[0, 1])
A17.description = """MCAT and GPAs for Applicants and Matriculants to U.S. Medical Schools by Primary Undergraduate Major, 2024-2025"""
output_path = os.path.join(output_dir, "Undergrad major MCAT Scores and GPAs.xlsx")
A17.to_excel(output_path)

#A23
A23 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-23.xlsx", 
                   header=[0,1])
A23.description = """MCAT and GPA Grid for Applicants and Acceptees to U.S. Medical Schools, 2022-2023 through 2024-2025 (Aggregated)"""
output_path = os.path.join(output_dir, "MCAT GPA Grid.xlsx")
A23.to_excel(output_path)

# A24
A24 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-24.xlsx", 
                   header=[0,1])
A24.description = """Applicants, Acceptees, and Matriculants to U.S. MD-Granting Medical Schools by Socioeconomic Status (SES), 2018-2019 through 2024-2025
"EO1/2" includes applicants whose parent(s) completed highest level of education was less than a Bachelor's degree or any degree with a service, clerical, skilled and unskilled occupation. 
"EO/3/4/5" includes applicants who parent(s) completed highest level of education was a Bachelor's, Master's, or Doctoral degree with an executive, managerial, or professional occupation. 
"Not Applicable" includes applicants whose parent(s) completed highest level of education outside of the U.S. and are not legal residents of the U.S.; parent(s) deceased; no parent data; or, applicant is not a U.S. citizen or permanent resident. 
"Unknown" includes applicants where all parental EO levels are “Unknown,” one parental EO level is “Unknown” and all other parental EO levels are “Not Applicable,” or applicants provided no parental information. 
SES is not calculated for non-U.S. citizens and non-permanent residents."""
output_path = os.path.join(output_dir, "SocioEconomic.xlsx")
A24.to_excel(output_path)

# A26
A26 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-26.xlsx", header=0)
A26.description = """First Generation Applicants, Acceptees, and Matriculants to U.S. MD-Granting Medical Schools, 2020-2021 through 2024-2025"""
A26 = A26.dropna(axis=0, how='all')
A26.iloc[:,0] = A26.iloc[:,0].ffill()
output_path = os.path.join(output_dir, "FirstGen.xlsx")
A26.to_excel(output_path)

# A27
A27 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\2024_FACTS_Table_A-27.xlsx", header=0)
A27.description = """Applicants, Acceptees, and Matriculants to U.S. MD-Granting Medical Schools who Applied Fee Assistance Benefits to AMCAS Applications, 2022-2023 through 2024-2025"""
output_path = os.path.join(output_dir, "FeeAssistance.xlsx")
A27.to_excel(output_path)

# charts
chart1 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\Chart 1 - Applicants, First-Time Applicants, Repeat Applicants to U.S. Medical Schools, 1980-1981 through 2024-2025.xlsx", 
                              engine=None, 
                              sheet_name=1,
                              header=3, 
                              na_filter=True, 
                              na_values=[" "])
chart1.description = """Applicants, First-Time Applicants, Repeat Applicants to U.S. Medical Schools, 1980-1981 through 2024-2025"""

chart2 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\Chart 2 - Applicants to U.S. Medical Schools by Gender, 1980-1981 through 2024-2025_0.xlsx", 
                              engine=None, 
                              sheet_name=1,
                              header=3, 
                              na_filter=True, 
                              na_values=[" "])
chart2.description = """Applicants to U.S. Medical Schools by Gender, 1980-1981 through 2024-2025"""

chart3 = pd.read_excel(r"C:\Users\yli355\Downloads\analysis\Chart 3 - Matriculants to U.S. Medical Schools by Gender, 1980-1981 through 2024-2025.xlsx", 
                              engine=None, 
                              sheet_name=1,
                              header=3, 
                              na_filter=True, 
                              na_values=[" "])
chart3.description = """Matriculants to U.S. Medical Schools by Gender, 1980-1981 through 2024-2025
Note: Matriculants who reported "Another Gender Identity" and did not report gender are only reflected in the “All Matriculants” counts."""
chart3 = chart3.iloc[:-1]

output_path = os.path.join(output_dir, "Rates over the years.xlsx")

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    chart1.to_excel(writer, sheet_name="Applicant type")
    chart2.to_excel(writer, sheet_name="Gender (Applicant)")
    chart3.to_excel(writer, sheet_name="Gender (Matriculant)")



