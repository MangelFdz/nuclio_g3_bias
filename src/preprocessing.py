

def impute_missing_col1(df):
    return df

def impute_missing_col2(df):
    return df

def impute_missing(df):
    df = impute_missing_col1(df)
    df = impute_missing_col2(df)
    return df