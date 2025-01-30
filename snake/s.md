## Snake Game - Aprendizado por reforço

O aprendizado por reforço é uma técnica de aprendizado de máquina que visa ensinar um agente a tomar decisões a partir de um estado e um  conjunto de ações possíveis, num determinado ambiente. Inicialmente, ele opta por aleatoriedade, mas com o tempo, ele aprende a tomar decisões melhores, baseadas em recompensas e punições.

Ou seja, num espectro filosófico, o Aprendizado por reforço simula o comportamento de um ser vivo. Por exemplo, um cachorro aprende a sentar quando recebe um petisco, e aprende a não latir quando é repreendido. 

O aprendizado por reforço é composto, inicialmente, por três elementos:
- **Agente**: O agente é o ser que toma as decisões. No nosso caso, o agente é a cobra.
- **Ambiente**: O ambiente é o local onde o agente toma as decisões. No nosso caso, o ambiente é o jogo da cobra.
- **Estado**: O estado é a situação atual do ambiente. No nosso caso, o estado é a posição da cobra, a posição da comida e a posição dos obstáculos.

Nesse projeto, o objetivo é ensinar um agente a jogar o jogo da cobrinha (snake) utilizando aprendizado por reforço.

### Implementando o jogo

Inicialmente, o jogo é implementado de forma simples usando o pygame. O jogo trabalha numa janela 800x600, onde cada célula do jogo é um quadrado de 20x20 pixels; Logo, o jogo pode ser interpretado como uma matriz 40x30.

A cobra possui o tamanho, que começa com 3 e aumenta conforme ela come a maçã, além disso ela possui uma direção. Resumindo os atributos da cobra:
- x: posição x da cabeça que varia de 0 a 800 (ou 0 a 39)
- y: posição y da cabeça que varia de 0 a 600 (ou 0 a 29)
- dx: direção x, que varia de -20 a 20 (ou -1,0,1)
- dy: direção y, que varia de -20 a 20 (ou -1,0,1)
- length: tamanho da cobra, que varia de 1 a N

Com a direção, é possível mapear a posição da cobra na matriz 40x30. Por exemplo, se a cobra está na posição (100,10) e a direção é (20,0), a próxima posição da cobra será (100-L,10), onde L é o tamanho da cobra. O sinal é negativo pois se a cobra se move para a direita, sua cabeça é a última célula da esquerda pra direita.

![alt text](body.png)

Dessa forma, podemos mapear a posição da cobra na matriz 40x30. Por consequinte, temos a nossa estrutura que representará o estado do jogo. Em cada célula da matriz, teremos:
- 0: célula vazia
- 1: corpo da cobra
- 2: cabeça da cobra
- 3: comida
  
Trabalhando nessa espécie de tabuleiro, também poderemos, posteriormente, adicionar obstáculos e mais coisas.

### Projetando o aprendizado

Bem, queremos que nossa AISnake aprenda a jogar bem o jogo da cobrinha, ou seja, de forma otimizada, e isso quer dizer que ela deve se alimentar o mais rápido possível e não morrer. Para isso, precisamos definir as recompensas e punições que a cobra receberá.

#### Recompensas
- Comer a comida: +10
- Morrer: -10
- Se mover: -0.1
  
#### Jogo
O jogo troca de época quando a cobra morre ou quando uma determinada quantidade de movimentos é feita. 

Essa quantidade aumenta a cada 100 épocas. Dessa forma, conforme a cobra aprende, ela terá mais tempo para jogar e, consequentemente, mais tempo para aprender.

#### Implementação

