import torch
import platform
import sys
import locale

def check_system_config():
    # Configura a codificação para UTF-8
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Verifica versão do PyTorch
    print(f"Versão do PyTorch: {torch.__version__}")
    
    # Verifica disponibilidade de CUDA
    cuda_disponivel = torch.cuda.is_available()
    print(f"\nCUDA Disponível: {cuda_disponivel}")
    
    # Informações sobre GPU
    if cuda_disponivel:
        print(f"Total de GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Nenhuma GPU encontrada - Executando em modo CPU")
    
    # Informações do sistema
    print(f"\nSistema Operacional: {platform.system()} {platform.version()}")
    print(f"Versão do Python: {sys.version.split()[0]}")
    print(f"Codificação do Sistema: {locale.getpreferredencoding()}")

if __name__ == "__main__":
    check_system_config()
