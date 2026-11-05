import pandas as pd
df_importado = pd.read_csv("analisis.csv")

resultado = df_importado.loc[0:19, ["estudiante", "horas_estudio"]]

print(resultado)
