#!/bin/bash

echo "Iniciando setup do projeto..."

# Verificar se Python esta instalado
if ! command -v python3 &> /dev/null; then
    echo "Python nao encontrado! Por favor, instale Python 3.8 ou superior."
    exit 1
fi

# Criar ambiente virtual
echo "Criando ambiente virtual..."
python3 -m venv venv

# Ativar ambiente virtual
echo "Ativando ambiente virtual..."
source venv/bin/activate

# Instalar dependencias
echo "Instalando dependencias..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Setup concluido com sucesso!"
echo "Para iniciar o programa, use um dos comandos:"
echo "- streamlit run streamlit_inference.py"
echo "- python detectar_usando_webcam.py"
echo "- python detectar_capturando_tela.py"

# Dar permissão de execução ao script
chmod +x setup.sh 