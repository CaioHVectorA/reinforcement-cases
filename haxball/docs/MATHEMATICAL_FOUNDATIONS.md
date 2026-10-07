# Fundamentos Matemáticos e Derivações Físicas: HaxBall RL

> Este documento estabelece o formalismo matemático rigoroso do motor de física 2D, a formulação de Processos de Decisão Markovianos Parcialmente Observáveis Descentralizados (Dec-POMDP), as derivações do estimador de vantagem GAE-$\\lambda$, a convergência do PPO com gradientes cortados e a prova de invariância por permutação via Auto-Atenção.

---

## 1. Mecânica Clássica e Dinâmica de Corpos Rígidos 2D

O simulador implementa integração numérica simplética de corpos rígidos no plano bidimensional $\mathbb{R}^2$ a uma frequência determinística de $f = 60\,\text{Hz}$ ($\Delta t = 1/60\,\text{s}$).

### 1.1. Integrador Simplético de Euler com Amortecimento Viscoso

Para um disco $i$ com posição $\vec{p}_i \in \mathbb{R}^2$, velocidade $\vec{v}_i \in \mathbb{R}^2$, aceleração de entrada $\vec{u}_i \in [-1, 1]^2$, constante de aceleração $a_i \in \mathbb{R}^+$, e coeficiente de amortecimento viscoso $d_i \in (0, 1]$:

$$\vec{v}_{i, t+1} = \left( \vec{v}_{i, t} + \vec{u}_{i, t} \cdot a_i \right) \cdot d_i$$
$$\vec{p}_{i, t+1} = \vec{p}_{i, t} + \vec{v}_{i, t+1}$$

* **Propriedade Simplética:** O esquema atualiza a velocidade antes da posição ($\text{Euler-Cromer}$), preservando a estabilidade da energia do sistema em oscilações e prevenindo divergência numérica em colisões elásticas de alta frequência.

---

### 1.2. Resolução de Colisão Disco-Disco (Momento Linear e Restituição)

Considere dois discos $A$ e $B$ com massas $m_A, m_B$ (e massas inversas $w_A = 1/m_A, w_B = 1/m_B$), raios $r_A, r_B$ e coeficientes de restituição $b_A, b_B$.

1. **Condição de Penetração:**
   $$\vec{\Delta} = \vec{p}_A - \vec{p}_B, \quad \text{dist} = \|\vec{\Delta}\|, \quad r_{\text{sum}} = r_A + r_B$$
   Existe colisão se e somente se:
   $$\delta = r_{\text{sum}} - \text{dist} > 0$$

2. **Vetor Normal de Contato:**
   $$\hat{n} = \frac{\vec{\Delta}}{\|\vec{\Delta}\|}$$

3. **Correção Posicional Instantânea (Anti-Interpenetração):**
   $$\vec{p}_A \leftarrow \vec{p}_A + \hat{n} \cdot \left(\delta \cdot \frac{w_A}{w_A + w_B}\right)$$
   $$\vec{p}_B \leftarrow \vec{p}_B - \hat{n} \cdot \left(\delta \cdot \frac{w_B}{w_A + w_B}\right)$$

4. **Velocidade Relativa Normal e Impulso:**
   A velocidade relativa projetada no vetor normal é:
   $$v_{\text{rel}} = (\vec{v}_A - \vec{v}_B) \cdot \hat{n}$$
   Se $v_{\text{rel}} < 0$ (discos se aproximando), o coeficiente de restituição efetivo é $e = b_A \times b_B$. O impulso escalar $J$ transferido satisfaz a conservação do momento linear:
   $$J = \frac{-(1 + e) v_{\text{rel}}}{w_A + w_B}$$
   As velocidades pós-colisão são:
   $$\vec{v}_A \leftarrow \vec{v}_A + \hat{n} \cdot (J \cdot w_A)$$
   $$\vec{v}_B \leftarrow \vec{v}_B - \hat{n} \cdot (J \cdot w_B)$$

---

### 1.3. Dinâmica de Ricochete na Parede (Segmentos e Reflexão Óptica)

Para um segmento retilíneo de parede definido pelos vértices $V_0, V_1 \in \mathbb{R}^2$:

1. **Vetor Diretor e Projeção:**
   $$\vec{d} = V_1 - V_0, \quad L^2 = \|\vec{d}\|^2$$
   O parâmetro de projeção do centro do disco $\vec{p}$ sobre o segmento é:
   $$t = \text{clip}\left( \frac{(\vec{p} - V_0) \cdot \vec{d}}{L^2}, \, 0.0, \, 1.0 \right)$$
   O ponto de contato mais próximo é:
   $$\vec{q} = V_0 + t \cdot \vec{d}$$

2. **Vetor de Reflexão Elástica:**
   Sendo $\hat{n} = \frac{\vec{p} - \vec{q}}{\|\vec{p} - \vec{q}\|}$ a normal da parede e $e = b_{\text{parede}} \times b_{\text{disco}}$:
   $$\vec{v}' = \vec{v} - (1 + e)(\vec{v} \cdot \hat{n})\hat{n}$$

* **Teorema da Tabela Master:** Se a parede é paralela ao eixo $X$ ($\hat{n} = (0, \pm 1)$) e $e = 1.25$ (quadra de futsal):
  $$v_x' = v_x \cdot d_{\text{damping}}$$
  $$v_y' = -1.25 \cdot v_y \cdot d_{\text{damping}}$$
  A componente perpendicular ganha aceleração elástica de $25\%$ no ricochete, permitindo passes rápidos que ultrapassam a marcação defensiva direta.

---

### 1.4. Mecânica do Chute e Inelasticidade de Condução

1. **Chute (*Kick Impulse*):**
   Quando a tecla de chute é ativada e $\|\vec{p}_{\text{bola}} - \vec{p}_{\text{jogador}}\| \le r_j + r_b + \text{kickMargin}$:
   $$\hat{k} = \frac{\vec{p}_{\text{bola}} - \vec{p}_{\text{jogador}}}{\|\vec{p}_{\text{bola}} - \vec{p}_{\text{jogador}}\|}$$
   $$\vec{v}_{\text{bola}} \leftarrow \vec{v}_{\text{bola}} + \hat{k} \cdot \text{kickStrength}$$

2. **Condução Inelástica de Futsal ($bCoef = 0.0$):**
   Quando a bola toca o jogador sem chute ativo, $e = 0.0 \times 0.5 = 0.0$.
   $$J = \frac{-v_{\text{rel}}}{w_A + w_B}$$
   A velocidade relativa na normal anula-se completamente ($v_{\text{rel}}' = 0$), fazendo com que a bola permaneça acoplada à trajetória do jogador (*dribble tapping* contínuo).

---

## 2. Formulação Dec-POMDP e Jogos Estocásticos de Markov

Modelamos uma partida $N$ vs $N$ como um **Jogo de Markov Parcialmente Observável**:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{I}, \{\mathcal{A}_i\}_{i \in \mathcal{I}}, \mathcal{P}, \{\mathcal{R}_i\}_{i \in \mathcal{I}}, \{\Omega_i\}_{i \in \mathcal{I}}, \gamma \rangle$$

### 2.1. Função de Valor de Estado e Equação de Bellman
Para uma política conjunta $\boldsymbol{\pi} = (\pi_1, \dots, \pi_{2N})$, o valor esperado para o time $\mathcal{T}$ a partir do estado $s$ é:
$$V^{\boldsymbol{\pi}}(s) = \mathbb{E}_{\boldsymbol{\pi}} \left[ \sum_{t=0}^{\infty} \gamma^t R_{\mathcal{T}}(s_t, \mathbf{a}_t) \,\middle|\, s_0 = s \right]$$

Sob a Equação de Bellman:
$$V^{\boldsymbol{\pi}}(s) = \sum_{\mathbf{a}} \boldsymbol{\pi}(\mathbf{a} \mid s) \left[ R_{\mathcal{T}}(s, \mathbf{a}) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, \mathbf{a}) V^{\boldsymbol{\pi}}(s') \right]$$

---

## 3. Generalized Advantage Estimation (GAE-$\lambda$)

O estimador GAE-$\lambda$ balanceia o viés (*bias*) do bootstrap do Crítico e a variância (*variance*) do retorno de Monte Carlo:

$$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$

* **Caso Limite $\lambda = 0$:**
  $$\hat{A}_t^{\text{GAE}(\gamma, 0)} = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$
  (Mínima variância, porém viés condicionado à precisão de $V$).
* **Caso Limite $\lambda = 1$:**
  $$\hat{A}_t^{\text{GAE}(\gamma, 1)} = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t) = G_t - V(s_t)$$
  (Estimador não-viesado de Monte Carlo, porém com variância máxima).
* **Configuração Ótima HaxBall:** $\lambda = 0.95, \gamma = 0.99$ reduz a variância estocástica de gols esparsos preservando sinais densos de passe e aproximação.

---

## 4. Otimização por Política Próxima (PPO-Clip)

A perda do PPO previne atualizações de política destrutivas através do corte do quociente de probabilidades:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

A função de perda multiobjetivo completa minimizada pelo Adam é:
$$\mathcal{L}(\theta) = -L^{\text{CLIP}}(\theta) + c_1 L^{\text{VF}}(\theta) - c_2 \mathcal{H}(\pi_\theta)$$
onde:
* $L^{\text{VF}}(\theta) = \hat{\mathbb{E}}_t \left[ \left( V_\theta(s_t) - \hat{R}_t \right)^2 \right]$ é o erro quadrático médio da função de valor.
* $\mathcal{H}(\pi_\theta) = \hat{\mathbb{E}}_t \left[ \sum_i \log \left(\sqrt{2\pi e}\,\sigma_i\right) \right]$ é a entropia diferencial da distribuição Gaussiana, incentivando a exploração contínua.
* $c_1 = 0.5, \, c_2 = 0.01, \, \epsilon = 0.20$.

---

## 5. Prova de Invariância por Permutação via Multi-Head Self-Attention

### 5.1. Definição Formal de Permutação
Seja $\mathcal{P}_K$ o grupo simétrico de permutações sobre $K$ entidades de entrada. Seja $P \in \mathbb{R}^{K \times K}$ a matriz de permutação correspondente.

### 5.2. Teorema da Equivariância por Permutação da Atenção
Uma camada de Auto-Atenção $\text{Attn}(X) = \text{softmax}\left(\frac{X W_Q W_K^T X^T}{\sqrt{d}}\right) X W_V$ é **permutação-equivariante**:
$$\text{Attn}(P X) = P \cdot \text{Attn}(X), \quad \forall P \in \mathcal{P}_K$$

**Demonstração:**
1. A matriz de scores sob permutação $P$ torna-se:
   $$S = \frac{(P X W_Q)(P X W_K)^T}{\sqrt{d}} = P \left(\frac{X W_Q W_K^T X^T}{\sqrt{d}}\right) P^T$$
2. Como a função $\text{softmax}$ atua ao longo das linhas e $P$ é uma matriz ortogonal de permutação:
   $$\text{softmax}(P A P^T) = P \cdot \text{softmax}(A) \cdot P^T$$
3. Multiplicando pelos valores $V = P X W_V$:
   $$\text{Attn}(P X) = \left( P \cdot \text{softmax}(A) \cdot P^T \right) (P X W_V) = P \cdot \text{softmax}(A) \cdot (P^T P) X W_V$$
   Como $P^T P = I$:
   $$\text{Attn}(P X) = P \cdot \text{Attn}(X) \quad \blacksquare$$

Ao aplicar um pooling simétrico (ex.: soma ou média) após o bloco de atenção, a política resultante torna-se estritamente **invariante por permutação**, garantindo que a ordem em que os jogadores são listados na observação não altere a probabilidade das ações.
