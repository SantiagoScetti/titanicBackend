import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import pprint

# Ruta específica para el proyecto TITANIC_API
MODEL_PATH = Path("models") / "modelo_titanic_vc2.pkl"

def analizar_modelo_titanic():
    """
    Analiza específicamente el modelo del Titanic y muestra información
    relevante para el desarrollo de la API.
    """
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DEL MODELO TITANIC: {MODEL_PATH}")
    print(f"{'='*60}\n")
    
    try:
        # Cargar el modelo
        print(f"Cargando modelo desde: {MODEL_PATH}")
        modelo = joblib.load(MODEL_PATH)
        print("✅ Modelo cargado exitosamente\n")
        
        # Información básica
        print(f"Tipo de modelo: {type(modelo).__name__}")
        
        # Analizar estructura del VotingClassifier
        if hasattr(modelo, 'estimators'):
            print("\nEstimadores del VotingClassifier:")
            for nombre, estimador in modelo.estimators:
                print(f"  - {nombre}: {type(estimador).__name__}")
                
                # Si es un Pipeline, analizar sus pasos
                if hasattr(estimador, 'steps'):
                    print(f"    Pasos del Pipeline {nombre}:")
                    for i, (nombre_paso, paso) in enumerate(estimador.steps):
                        print(f"      {i+1}. {nombre_paso}: {type(paso).__name__}")
        
        # Intentar determinar las columnas y características esperadas
        print("\nAnalizando características esperadas...")
        
        # Crear un pequeño DataFrame de prueba con datos similares a la petición de Postman
        test_data = pd.DataFrame({
            "name": ["Test Person"],
            "Pclass": [1],
            "Sex": ["male"],
            "Age": [30],
            "SibSp": [1],
            "Parch": [0],
            "Fare": [71.3],
            "Embarked": ["S"],
            "Cabin_Assigned": [0],
            "Name_Size": [10],
            "TicketNumberCounts": [2],
            "Family_Size_Grouped": ["Small"]
        })
        
        print("\nEstructura de los datos de prueba:")
        print(f"Columnas: {test_data.columns.tolist()}")
        print(f"Número de características: {test_data.shape[1]}")
        
        # Intentar hacer una predicción para ver qué error obtenemos
        print("\nIntentando predicción con datos de prueba...")
        try:
            prediction = modelo.predict(test_data)
            print("✅ Predicción exitosa")
            print(f"Resultado: {prediction}")
        except Exception as e:
            print(f"❌ Error en la predicción: {str(e)}")
            
            # Si hay error por número de características, intentar determinar cuántas espera
            if "features" in str(e) and "expecting" in str(e):
                error_msg = str(e)
                import re
                match = re.search(r'(\d+) features, but .+ is expecting (\d+)', error_msg)
                if match:
                    enviadas, esperadas = match.groups()
                    print(f"\nEnviaste {enviadas} características, pero el modelo espera {esperadas}")
                    print("Necesitamos reconciliar esta diferencia.")
            
            # Probar añadiendo Family_Size
            print("\nProbando con Family_Size = SibSp + Parch")
            test_data['Family_Size'] = test_data['SibSp'] + test_data['Parch']
            try:
                prediction = modelo.predict(test_data)
                print("✅ Predicción exitosa al añadir Family_Size")
                print(f"Resultado: {prediction}")
            except Exception as e:
                print(f"❌ Sigue fallando: {str(e)}")
        
        # Probar con varias combinaciones de columnas
        print("\nProbando combinaciones de columnas comunes...")
        column_sets = [
            ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Family_Size_Grouped', 
             'Cabin_Assigned', 'Name_Size', 'TicketNumberCounts', 'Family_Size'],
            
            ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
             'Family_Size_Grouped', 'Cabin_Assigned', 'Name_Size', 'TicketNumberCounts'],
            
            ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Family_Size_Grouped', 
             'Family_Size', 'Cabin_Assigned', 'Name_Size', 'TicketNumberCounts', 'Title'],
        ]
        
        for i, cols in enumerate(column_sets):
            # Verificar qué columnas del conjunto están en nuestros datos de prueba
            missing = [col for col in cols if col not in test_data.columns]
            if missing:
                print(f"\nConjunto {i+1}: Faltan columnas: {missing}")
                continue
                
            test_subset = test_data[cols]
            print(f"\nProbando conjunto {i+1} con columnas: {cols}")
            try:
                prediction = modelo.predict(test_subset)
                print(f"✅ ¡ÉXITO! El modelo funciona con este conjunto de {len(cols)} columnas")
                print(f"Columnas: {cols}")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        # Verificar la generación de probabilidades
        print("\nProbando generación de probabilidades...")
        try:
            if hasattr(modelo, 'predict_proba'):
                proba = modelo.predict_proba(test_subset)
                print("✅ predict_proba disponible y funciona")
                print(f"Probabilidades: {proba}")
            else:
                print("❌ El modelo no tiene método predict_proba")
                
                # Probar con el método de votos
                print("\nProbando cálculo manual de probabilidades mediante votos...")
                try:
                    individual_preds = [est.predict(test_subset)[0] for _, est in modelo.estimators]
                    votes = sum(individual_preds)
                    probability = float(votes / len(modelo.estimators))
                    print("✅ Cálculo por votos funciona")
                    print(f"Votos positivos: {votes}/{len(modelo.estimators)}")
                    print(f"Probabilidad: {probability}")
                except Exception as e:
                    print(f"❌ Error calculando por votos: {str(e)}")
        except Exception as e:
            print(f"Error probando probabilidades: {str(e)}")
        
        # Recomendaciones finales
        print(f"\n{'='*30} RECOMENDACIONES {'='*30}")
        print("1. Asegúrate de incluir la columna 'Family_Size' = SibSp + Parch")
        print("2. Usa exactamente las columnas en el mismo orden que funcionaron en las pruebas")
        print("3. Si las probabilidades siempre son 0 o 1, usa el método de votos implementado aquí")
        print("4. Considera entrenar un nuevo modelo que incluya el pipeline completo")
        
    except Exception as e:
        print(f"❌ ERROR CRÍTICO: {str(e)}")

if __name__ == "__main__":
    analizar_modelo_titanic()