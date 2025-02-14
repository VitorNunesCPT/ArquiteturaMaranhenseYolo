def verificar_rotulos(log_file, rotulos):
    rotulos_encontrados = []

    try:
        with open(log_file, 'r') as f:
            conteudo = f.read()

            for rotulo in rotulos:
                if rotulo in conteudo:
                    rotulos_encontrados.append(rotulo)

    except FileNotFoundError:
        print(f"Arquivo de log '{log_file}' não encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

    return rotulos_encontrados


# Lista de rótulos para verificar
rotulos = [
    "poste colonial",
    "grade ornamental colonial",
    "rua de pedra colonial",
    "porta arquitetura colonial",
    "igreja da Se",
    "janela colonial",
    "escultura leao heraldica"
]

# Arquivo de log a ser analisado
log_file = "terminal.log"

# Verificar rótulos no log
rotulos_encontrados = verificar_rotulos(log_file, rotulos)

# Exibir resultados
if rotulos_encontrados:
    print("Rotulos encontrados no log:")
    for rotulo in rotulos_encontrados:
        print(f"- {rotulo}")
else:
    print("Nenhum dos rótulos especificados foi encontrado no log.")
