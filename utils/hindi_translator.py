from googletrans import Translator

def translate_dataframe_to_hindi(df):
    translator = Translator()
    df_translated = df.copy()

    # Translate column names
    translated_cols = {}
    for col in df.columns:
        try:
            translated_cols[col] = translator.translate(col, dest="hi").text
        except Exception as e:
            translated_cols[col] = col  # fallback
    df_translated.rename(columns=translated_cols, inplace=True)

    # Translate all string/object values
    for col in df_translated.columns:
        if df_translated[col].dtype == "object":
            try:
                df_translated[col] = df_translated[col].astype(str).apply(lambda x: translator.translate(x, dest="hi").text)
            except Exception as e:
                pass  # silently fail for any column

    return df_translated
