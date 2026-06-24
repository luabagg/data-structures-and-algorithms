import numpy as np
import matplotlib.pyplot as plt

def linear_least_squares(x, y):
    """
    Implementação do método dos quadrados mínimos para ajuste linear.
    y = a + bx
    
    Parameters:
    x (array): Valores do eixo x
    y (array): Valores do eixo y
    
    Returns:
    tuple: (a, b) coeficientes da reta, soma_residuos_quadraticos
    """
    n = len(x)
    
    # Passo 1: Calcular somas necessárias
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
    sum_x2 = sum(x_i**2 for x_i in x)
    
    # Passo 2: Montar o sistema de equações normais
    # Sistema:
    # a*n + b*sum_x = sum_y
    # a*sum_x + b*sum_x2 = sum_xy
    
    # Passo 3: Resolver o sistema para encontrar a e b
    denominator = n * sum_x2 - sum_x**2
    
    if denominator == 0:
        raise ValueError("O sistema não tem solução única.")
    
    a = (sum_y * sum_x2 - sum_x * sum_xy) / denominator
    b = (n * sum_xy - sum_x * sum_y) / denominator
    
    # Passo 4: Calcular os valores ajustados
    y_fitted = [a + b * x_i for x_i in x]
    
    # Passo 5: Calcular os resíduos
    residuals = [y_i - y_fit_i for y_i, y_fit_i in zip(y, y_fitted)]
    
    # Passo 6: Calcular a soma dos quadrados dos resíduos
    sum_squared_residuals = sum(res**2 for res in residuals)
    
    return a, b, sum_squared_residuals

def polynomial_least_squares(x, y, degree=2):
    """
    Implementação do método dos quadrados mínimos para ajuste polinomial.
    y = a_0 + a_1*x + a_2*x^2 + ... + a_n*x^n
    
    Parameters:
    x (array): Valores do eixo x
    y (array): Valores do eixo y
    degree (int): Grau do polinômio
    
    Returns:
    array: Coeficientes do polinômio (do termo constante até o maior grau)
    float: Soma dos quadrados dos resíduos
    """
    # Usando numpy para resolver o sistema de equações normais
    A = np.vander(x, degree+1, increasing=True)
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Calcular os valores ajustados
    y_fitted = np.polyval(coeffs[::-1], x)
    
    # Calcular a soma dos quadrados dos resíduos
    residuals = y - y_fitted
    sum_squared_residuals = np.sum(residuals**2)
    
    return coeffs, sum_squared_residuals

def plot_results(x, y, model_func, params, title):
    """
    Plota os dados originais e a curva ajustada.
    
    Parameters:
    x (array): Valores do eixo x
    y (array): Valores do eixo y
    model_func (function): Função que define o modelo ajustado
    params (tuple/array): Parâmetros do modelo
    title (str): Título do gráfico
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Dados originais')
    
    # Gerar pontos para plotar a curva ajustada
    x_line = np.linspace(min(x), max(x), 100)
    y_line = model_func(x_line, params)
    
    plt.plot(x_line, y_line, color='red', label='Curva ajustada')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def linear_model(x, params):
    """Modelo linear y = a + bx"""
    a, b = params
    return a + b * x

def polynomial_model(x, params):
    """Modelo polinomial"""
    return np.polyval(params[::-1], x)

def demonstrate_least_squares():
    """Demonstração do método dos quadrados mínimos com dados de exemplo"""
    # Exemplo 1: Dados com tendência linear + ruído
    np.random.seed(42)
    x1 = np.linspace(0, 10, 20)
    y1 = 2 + 3 * x1 + np.random.normal(0, 1.5, len(x1))
    
    print("=== Exemplo de Regressão Linear ===")
    a, b, ssr = linear_least_squares(x1, y1)
    print(f"Coeficientes encontrados: a = {a:.4f}, b = {b:.4f}")
    print(f"Equação da reta: y = {a:.4f} + {b:.4f}x")
    print(f"Soma dos quadrados dos resíduos: {ssr:.4f}")
    
    plot_results(x1, y1, linear_model, (a, b), "Ajuste Linear pelo Método dos Quadrados Mínimos")
    
    # Exemplo 2: Dados com tendência quadrática + ruído
    x2 = np.linspace(-5, 5, 30)
    y2 = 1 + 0.5 * x2 + 1.3 * x2**2 + np.random.normal(0, 3, len(x2))
    
    print("\n=== Exemplo de Regressão Polinomial (Grau 2) ===")
    coeffs, ssr = polynomial_least_squares(x2, y2, degree=2)
    print(f"Coeficientes encontrados: {coeffs}")
    print(f"Equação do polinômio: y = {coeffs[0]:.4f} + {coeffs[1]:.4f}x + {coeffs[2]:.4f}x²")
    print(f"Soma dos quadrados dos resíduos: {ssr:.4f}")
    
    plot_results(x2, y2, polynomial_model, coeffs, "Ajuste Polinomial pelo Método dos Quadrados Mínimos")

if __name__ == "__main__":
    demonstrate_least_squares()