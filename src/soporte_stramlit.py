import pickle
import pandas as pd
import numpy as np

with open('../encoders/target_encoding.pkl', 'rb') as f:
    target_encoder = pickle.load(f)
with open('../encoders/one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)
with open('../encoders/robus_scaler.pkl', 'rb') as f:
    robust_scaler = pickle.load(f)    
with open('../models/mejor_modelo_random_forest.pkl', 'rb') as f:
    modelo = pickle.load(f)


def realizar_prediccion(EnvironmentSatisfaction,WorkLifeBalance,JobInvolvement,JobSatisfaction,
                        Age,Gender,MaritalStatus,DistanceFromHome,Department,JobLevel,JobRole,
                        YearsAtCompany,YearsSinceLastPromotion,YearsWithCurrManager,Education,
                        EducationField,TrainingTimesLastYear,MonthlyIncome,PercentSalaryHike,
                        StockOptionLevel,NumCompaniesWorked,TotalWorkingYears,BusinessTravel,PerformanceRating):
    """
    Realiza una predicción de la continuidad de un empleado basada en sus características utilizando un modelo entrenado.

    Parámetros:
    -----------
    EnvironmentSatisfaction : int
        Nivel de satisfacción con el entorno laboral (1-4, siendo 1 "Bajo" y 4 "Muy alto").
    WorkLifeBalance : int
        Nivel de equilibrio entre vida personal y laboral (1-4, siendo 1 "Bajo" y 4 "Muy alto").
    JobInvolvement : int
        Nivel de involucramiento en el trabajo (1-4, siendo 1 "Bajo" y 4 "Muy alto").
    JobSatisfaction : int
        Nivel de satisfacción con el trabajo (1-4, siendo 1 "Bajo" y 4 "Muy alto").
    Age : int
        Edad del empleado en años.
    Gender : str
        Género del empleado (e.g., "Male", "Female").
    MaritalStatus : str
        Estado civil del empleado (e.g., "Single", "Married", "Divorced").
    DistanceFromHome : int
        Distancia desde la casa al lugar de trabajo en kilómetros.
    Department : str
        Departamento en el que trabaja el empleado (e.g., "Sales", "HR", "R&D").
    JobLevel : int
        Nivel del puesto del empleado (1-5).
    JobRole : str
        Rol del trabajo del empleado (e.g., "Manager", "Sales Executive").
    YearsAtCompany : int
        Años que el empleado lleva en la compañía.
    YearsSinceLastPromotion : int
        Años desde la última promoción del empleado.
    YearsWithCurrManager : int
        Años que el empleado ha trabajado con su gerente actual.
    Education : int
        Nivel educativo del empleado (1-5).
    EducationField : str
        Campo de estudio del empleado (e.g., "Life Sciences", "Technical Degree").
    TrainingTimesLastYear : int
        Número de capacitaciones recibidas en el último año.
    MonthlyIncome : float
        Ingreso mensual del empleado.
    PercentSalaryHike : float
        Porcentaje de incremento salarial reciente del empleado.
    StockOptionLevel : int
        Nivel de opciones sobre acciones (0-3).
    NumCompaniesWorked : int
        Número de compañías en las que el empleado ha trabajado anteriormente.
    TotalWorkingYears : int
        Años totales de experiencia laboral del empleado.
    BusinessTravel : str
        Frecuencia de viajes de negocios (e.g., "Rarely", "Frequently").
    PerformanceRating : int
        Calificación de desempeño del empleado (1-4, siendo 4 "Sobresaliente").

    Retorna:
    --------
    numpy.ndarray
        Predicción del modelo para métricas relacionadas con el empleado, como probabilidad de rotación o desempeño esperado.

    Notas:
    ------
    - Las variables categóricas se dividen en:
        - `cols_target`: Variables codificadas con Target Encoding.
        - `cols_nominales`: Variables codificadas con OneHot Encoding.
        - `cols_escalar`: Variables numéricas que son escaladas.
    - El DataFrame `df_new` contiene las características del empleado y pasa por transformaciones antes de la predicción.

    Ejemplo:
    --------
    prediccion = realizar_prediccion(
        EnvironmentSatisfaction=3,
        WorkLifeBalance=4,
        JobInvolvement=3,
        JobSatisfaction=4,
        Age=35,
        Gender="Male",
        MaritalStatus="Married",
        DistanceFromHome=10,
        Department="R&D",
        JobLevel=2,
        JobRole="Research Scientist",
        YearsAtCompany=8,
        YearsSinceLastPromotion=2,
        YearsWithCurrManager=5,
        Education=3,
        EducationField="Life Sciences",
        TrainingTimesLastYear=2,
        MonthlyIncome=7000,
        PercentSalaryHike=12.5,
        StockOptionLevel=1,
        NumCompaniesWorked=3,
        TotalWorkingYears=12,
        BusinessTravel="Rarely",
        PerformanceRating=3,
        encoder_ordinales=target_encoder,
        encoder_nominales=one_hot_encoder,
        scaler=standard_scaler,
        modelo=modelo_rf
    )
    print(f"El precio estimado es: {prediccion[0]}")
    """
    cols_ordinales =['Department', 'EducationField', 'MaritalStatus']
    cols_nominales = ['Education', 'Gender', 'JobRole']
    cols_escalar = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
       'PercentSalaryHike', 'TotalWorkingYears', 'YearsAtCompany',
       'YearsSinceLastPromotion', 'YearsWithCurrManager']

    # Datos de una nueva casa para predicción

    new_employee = pd.DataFrame({
    'EnvironmentSatisfaction': [EnvironmentSatisfaction],
    'WorkLifeBalance': [WorkLifeBalance],
    'JobInvolvement': [JobInvolvement],
    'JobSatisfaction': [JobSatisfaction],
    'Age': [Age],
    'Gender': [Gender],
    'MaritalStatus': [MaritalStatus],
    'DistanceFromHome': [DistanceFromHome],
    'Department': [Department],
    'JobLevel': [JobLevel],
    'JobRole': [JobRole],
    'YearsAtCompany': [YearsAtCompany],
    'YearsSinceLastPromotion': [YearsSinceLastPromotion],
    'YearsWithCurrManager': [YearsWithCurrManager],
    'Education': [Education],
    'EducationField': [EducationField],
    'TrainingTimesLastYear': [TrainingTimesLastYear],
    'MonthlyIncome': [MonthlyIncome],
    'PercentSalaryHike': [PercentSalaryHike],
    'StockOptionLevel': [StockOptionLevel],
    'NumCompaniesWorked': [NumCompaniesWorked],
    'TotalWorkingYears': [TotalWorkingYears],
    'BusinessTravel': [BusinessTravel],
    'PerformanceRating': [PerformanceRating]})

    df_new = pd.DataFrame(new_employee)
    df_pred = df_new.copy()

    # Escalamos los valores
    df_pred[cols_escalar] = robust_scaler.transform(df_pred[cols_escalar])

    # Hacemos el OneHot Encoder
    onehot = one_hot_encoder.transform(df_pred[cols_nominales])
    # Obtenemos los nombres de las columnas del codificador
    column_names = one_hot_encoder.get_feature_names_out(cols_nominales)
    # Convertimos a un DataFrame
    onehot_df = pd.DataFrame(onehot.toarray(), columns=column_names)

    # Realizamos el target encoder
    df_pred["Attrition"] = np.nan #La creo porque la espera, luego se borra
    df_pred = target_encoder.transform(df_pred[cols_ordinales])

    # Quitamos las columnas que ya han sido onehoteadas 
    df_pred.drop(columns= cols_nominales,inplace=True)
    df_pred = pd.concat([df_pred, onehot_df], axis=1)

    # Realizamos la predicción
    df_pred.drop(columns="Attrition",inplace=True)
    prediccion = modelo.predict(df_pred)
    return prediccion