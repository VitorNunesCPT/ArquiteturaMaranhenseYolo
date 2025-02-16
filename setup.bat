@echo off
echo Iniciando setup do projeto...

:: Verificar se Python esta instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo Python nao encontrado! Por favor, instale Python 3.8 ou superior.
    exit /b 1
)

:: Criar ambiente virtual
echo Criando ambiente virtual...
python -m venv venv

:: Ativar ambiente virtual
echo Ativando ambiente virtual...
call venv\Scripts\activate

:: Instalar dependencias
echo Instalando dependencias...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Setup concluido com sucesso!
echo Para iniciar o programa, use um dos comandos:
echo - streamlit run streamlit_inference.py
echo - python detectar_usando_webcam.py
echo - python detectar_capturando_tela.py

pause 