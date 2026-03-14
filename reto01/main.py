import sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def clasificador_humano_v2(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    """
    Versión mejorada del clasificador humano.
    """
    # Regla 1: Usamos la precisión decimal del ML para separar Gentoo
    if flipper_length_mm > 206.5:
        return 'Gentoo'
    else:
        # Regla 2: Mejoramos la separación entre Adelie y Chinstrap
        if bill_length_mm <= 43.3:
            return 'Adelie'
        else:
            # Regla 3: Un último nivel de profundidad para casos ambiguos
            if bill_depth_mm > 16.5:
                return 'Chinstrap'
            else:
                return 'Adelie'

def main():
    if len(sys.argv) != 2:
        print("Para usar: python main.py <archivo.csv>")
        sys.exit(1)

    archivo_csv = sys.argv[1]

    try:
        # Cargar datos de entrenamiento originales (necesarios para el modelo ML)
        import seaborn as sns # Solo se usa rapido para cargar
        df_original = sns.load_dataset('penguins').dropna().reset_index(drop=True)
        X_train = df_original[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
        y_train = df_original['species']
        
        # Entrenar el modelo ML con los datos de Palmer original
        modelo_ml = DecisionTreeClassifier(random_state=42)
        modelo_ml.fit(X_train, y_train)

        # Cargar los datos nuevos desde el CSV proporcionado
        df = pd.read_csv(archivo_csv)
        
        columnas_necesarias = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        
        # Verificar que el CSV tenga las columnas necesarias
        if not all(col in df.columns for col in columnas_necesarias):
             print(f"Error: El archivo CSV debe contener las columnas {columnas_necesarias}")
             sys.exit(1)
        
        # Predicción Humano (Versión 2 por defecto ya que es mejor)
        predicciones_humano = []
        for idx, row in df.iterrows():
            pred = clasificador_humano_v2(
                row['bill_length_mm'],
                row['bill_depth_mm'],
                row['flipper_length_mm'],
                row['body_mass_g']
            )
            predicciones_humano.append(pred)

        # Predicción ML
        X_nuevos = df[columnas_necesarias]
        predicciones_ml = modelo_ml.predict(X_nuevos)

        if 'species' in df.columns:
            from sklearn.metrics import accuracy_score
            y_true = df['species']
            acc_humano = accuracy_score(y_true, predicciones_humano)
            acc_ml = accuracy_score(y_true, predicciones_ml)
            
            # Guardar el accuracy en un CSV
            resultados = pd.DataFrame({
                'prediccion_humano': [acc_humano],
                'prediccion_ml': [acc_ml]
            })
            
            output_filename = "resultados_accuracy.csv"
            resultados.to_csv(output_filename, index=False)
            print(f"Accuracy guardado exitosamente en {output_filename}")
            print(f"Accuracy Humano: {acc_humano:.2%}")
            print(f"Accuracy ML: {acc_ml:.2%}")
        else:
            print("El archivo CSV no contiene la columna 'species', por lo que no se puede calcular el accuracy.")


    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {archivo_csv}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()
