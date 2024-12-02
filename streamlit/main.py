# streamlit run main.py
import streamlit as st
import pandas as pd
import pickle

import sys
import os

# Ruta relativa desde notebooks/1-Preprocesing/ a src/
src_path = "../src/"
sys.path.append(src_path)
import soporte_stramlit as sp  

# Cargar las listas de opciones
lista_encuesta=[1,2,3,4]
lista_genero=['Female', 'Male']
lista_estado_civil=['Divorced', 'Married', 'Single']
lista_departamentos= ['Human Resources', 'Research & Development', 'Sales']
lista_job_role= ['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative']
lista_educacion=['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree']
lista_travel=['Non-Travel', 'Travel_Frequently', 'Travel_Rarely']
lista_level=[1,2,3,4,5]

# Configurar la página de Streamlit
st.set_page_config(
    page_title="Employee Predictor",
    page_icon="🧑🏾‍💼",
    layout="centered",
)

# Título y descripción
st.title("🧑🏾‍💼 Employee Predictor")
st.write("Usa esta aplicación para predecir si tu empleado esta mas fuera que dentro de la empresa. ¡Sorpréndete con la magia de los datos! 🚀")

# Mostrar una imagen llamativa
st.image(
    "https://plus.unsplash.com/premium_photo-1683120730432-b5ea74bd9047?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",  # URL de la imagen
    caption="Conoce si tu empleado abandonara o no la empresa",
    use_container_width=True,
)
st.write("Actualmente el modelo está limitado a seleccionar opciones sobre lo que ya conoce.")

# Formularios de entrada
st.header("¿Como es el empleado? 📑")

st.markdown("#### 1. Satisfacción y Balance")
col1,col2,col3,col4 = st.columns(4)
with col1:
    EnvironmentSatisfaction=st.selectbox("Satisfacción con el entorno:",lista_encuesta)
with col2:
    WorkLifeBalance = st.selectbox("Equilibrio entre trabajo y vida:",lista_encuesta)
with col3:
    JobInvolvement = st.selectbox("Involucración laboral:",lista_encuesta)
with col4:
    JobSatisfaction = st.selectbox("Satisfacción laboral:",lista_encuesta)

st.markdown("#### 2. Demografía y Datos Personales")
col5,col6,col7,col8 = st.columns(4)
with col5:
    Age = st.number_input("Edad:", min_value=18, max_value=60, value=42, step=1)
with col6:
    Gender = st.selectbox("Género:",lista_genero)
with col7:
    MaritalStatus = st.selectbox("Estado civil:",lista_estado_civil)
with col8:
    DistanceFromHome= st.number_input("Distancia desde el hogar:", min_value=1, max_value=29, value=28, step=1)

st.markdown("#### 3. Información Laboral")
col9,col10,col11 = st.columns(3)
with col9:
    Department = st.selectbox("Departamento:",lista_departamentos)
with col10:
    JobRole = st.selectbox("Rol laboral:",lista_job_role)
with col11:
    JobLevel = st.selectbox("Nivel de responsabilidad:",lista_level)

col12,col13,col14 = st.columns(3)
with col12:
    YearsAtCompany = st.number_input("Años en la compañía:", min_value=0, max_value=40, value=40, step=1)
with col13:
    YearsSinceLastPromotion = st.number_input("Años desde la última promoción:", min_value=0, max_value=15, value=15, step=1)
with col14:
    YearsWithCurrManager = st.number_input("Años con el gerente actual:", min_value=0, max_value=17, value=17, step=1)

st.markdown("#### 4. Formación y Educación")
col15,col16,col17 = st.columns(3)
with col15:
    Education = st.selectbox("Nivel educativo:", lista_level)
with col16:
    EducationField = st.selectbox("Área de educación:",lista_educacion)
with col17:
    TrainingTimesLastYear = st.number_input("Formaciones en el último año:",min_value=0, max_value=6, value=6, step=1)

st.markdown("#### 5. Compensación y Beneficios")
col18,col19,col20 = st.columns(3)
with col18:
    MonthlyIncome = st.number_input("Ingreso mensual:", min_value=10000, max_value=200000, value=19000, step=100)
with col19:
    PercentSalaryHike = st.number_input("Incremento porcentual de salario:",min_value=11, max_value=25, value=14, step=1)
with col20:
    StockOptionLevel = st.number_input("Nivel de opciones sobre acciones:",min_value=0, max_value=3, value=3, step=1)

st.markdown("#### 6. Experiencia Laboral")
col21,col22,col23 = st.columns(3)
with col21:
    TotalWorkingYears = st.number_input("Total años trabajados:",min_value=0, max_value=40, value=2, step=1)
with col22:
    NumCompaniesWorked = st.number_input("Número de empresas trabajadas:", min_value=0, max_value=9, value=2, step=1)
with col23:
    BusinessTravel = st.selectbox("Frecuencia de viajes:",lista_travel)

st.markdown("#### 7. Evaluación y Desempeño")
PerformanceRating = st.number_input("Calificación de desempeño:",min_value=3, max_value=4, value=3, step=1)

# Botón para realizar la predicción
if st.button("Predecir"):
    prediction = sp.realizar_prediccion(EnvironmentSatisfaction,WorkLifeBalance,JobInvolvement,JobSatisfaction,
                                        Age,Gender,MaritalStatus,DistanceFromHome,Department,JobLevel,JobRole,
                                        YearsAtCompany,YearsSinceLastPromotion,YearsWithCurrManager,Education,
                                        EducationField,TrainingTimesLastYear,MonthlyIncome,PercentSalaryHike,
                                        StockOptionLevel,NumCompaniesWorked,TotalWorkingYears,BusinessTravel,PerformanceRating)
    
    # Mostrar el resultado
    prediction=1
    if prediction==1:
        st.markdown("#### Se va de la empresa")

        st.image(
    "https://static.wixstatic.com/media/43d2ce_a123fc13f56d44dc964de2ca7ecc522c~mv2.jpg/v1/fill/w_755,h_396,al_c,lg_1,q_80,enc_auto/43d2ce_a123fc13f56d44dc964de2ca7ecc522c~mv2.jpg",  # URL de la imagen
    caption="",
    use_container_width=True,)
    if prediction==0:
        st.markdown("#### Se queda")

        st.image(
    "https://e00-expansion.uecdn.es/assets/multimedia/imagenes/2017/09/08/15048915173238.jpg",  # URL de la imagen
    caption="",
    use_container_width=True,)