# Engenharia Reversa e Especificação Técnica Completa: HaxBall Replay (.hbr2 / .hbr)
> **Documento Oficial de Referência de Engenharia Reversa, Estrutura Binária, Protocolo de Eventos e Ecossistema Global de Dados**  
> *Autor:* Equipe de Engenharia e Pesquisa HaxBall RL  
> *Versão:* 2.0 (Especificação HTML5 / HBR2)

---

## 1. Introdução e Visão Geral

O **HaxBall** armazena gravações de partidas nos formatos `.hbr` (legado da era Flash, 2010–2018) e **`.hbr2`** (formato oficial da versão HTML5/WebAssembly, 2018–presente).

Diferente de formatos convencionais de vídeo (MP4, WebM) ou demos de jogos baseadas em coordenadas contínuas de interpolação por frame (como em Source Engine ou Unreal), o `.hbr2` é um formato **puramente orientado a eventos e baseado em simulação determinística (*Event-Sourced Replay Architecture*)**:
1. **Armazena o estado inicial da sala e do mapa (stadium):** Geometria completa das paredes, gols, física dos discos e configurações de regras.
2. **Armazena apenas os deltas de inputs dos jogadores e eventos de rede:** A cada tick do relógio de física (60 Hz), apenas as teclas pressionadas, conexões/desconexões e gols são gravados.
3. **Reprodução determinística:** O cliente do HaxBall reconstrói a partida executando o motor de física (`e.T.O`) quadro a quadro a partir do estado inicial, garantindo arquivos compactos (geralmente entre 15 KB e 150 KB para partidas completas de 5 a 10 minutos).

---

## 2. Anatomia do Container e Camada de Transporte

O arquivo `.hbr2` é composto por um **cabeçalho binário fixo de 12 bytes** seguido por um **payload compactado com o algoritmo Deflate bruto**:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       'H'     |       'B'     |       'R'     |       '2'     |  Magic Bytes (4B)
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      File Version (uint32_be)                 |  Versão (4B)
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   Duration in Ticks (uint32_be)               |  Duração (4B)
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|        Payload Compactado com Raw Deflate (RFC 1951)          |  Offset 12+
|                   (wbits = -15 / pako.inflateRaw)             |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### 2.1. Campos do Cabeçalho Fixo (Offset 0..11)
* **Magic Bytes (Offset 0..3):** `0x48 0x42 0x52 0x32` (`b"HBR2"`). Identificador exclusivo do formato HTML5.
* **Versão do Formato (Offset 4..7):** Inteiro não assinado de 32 bits em Big-Endian (`>I`). Tipicamente `1` ou `2`.
* **Duração da Gravação (Offset 8..11):** Inteiro não assinado de 32 bits em Big-Endian (`>I`). Representa o total de **ticks lógicos** da física gravados. Como o motor roda a 60 ticks por segundo:
  $$\text{Duração (segundos)} = \frac{\text{Duration Ticks}}{60.0}$$

### 2.2. Algoritmo de Compressão do Payload (Offset 12 até o fim)
O payload não utiliza cabeçalhos ZLIB padrão (`0x78 0x9c`) nem containers GZIP. Trata-se de um **stream Deflate bruto (*Raw Deflate Stream*, RFC 1951)**.
* Em **Python**: deve ser descompactado via `zlib.decompress(payload, -15)`. (O argumento `-15` desativa a checagem de cabeçalhos ZLIB/Adler32).
* Em **Node.js / JavaScript**: utiliza-se `pako.inflateRaw(buffer)` ou `zlib.inflateRawSync(buffer)`.
* Em **Rust**: utiliza-se `flate2::read::DeflateDecoder`.

---

## 3. Primitivas Binárias e Codificação de Dados

O fluxo descompactado é lido sequencialmente por um leitor de bytes de baixo nível (*ByteReader*):

### 3.1. Primitivas Numéricas Padrão (Big-Endian)
* `u8` / `i8`: Bytes de 8 bits com/sem sinal.
* `u16_be` / `i16_be`: Inteiros de 16 bits em Big-Endian (`>H`, `>h`).
* `u32_be` / `i32_be`: Inteiros de 32 bits em Big-Endian (`>I`, `>i`).
* `f32_be`: Ponto flutuante IEEE 754 de precisão simples (4 bytes, `>f`).
* `f64_be`: Ponto flutuante IEEE 754 de precisão dupla (8 bytes, `>d`), amplamente utilizado para coordenadas de vértices e massas.

### 3.2. Codificação de Varints (LEB128 Modificado)
Para economizar espaço com números inteiros variáveis (como contadores de ticks e tamanhos de strings), o HaxBall adota uma variante do **LEB128 (Little-Endian Base 128)**:

```python
def read_varint(reader: ByteReader) -> int:
    result = 0
    shift = 0
    while True:
        byte = reader.read_u8()
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
    return result
```

### 3.3. Codificação de Strings
Existem dois formatos de strings no protocolo:
1. **String com Comprimento Varint (`str_varint_len`):**
   * Lê um `varint` indicando o comprimento em bytes $L$.
   * Lê os próximos $L$ bytes e decodifica como UTF-8.
2. **String Anulável (*Nullable String* - `nullable_str`):**
   * Lê um `varint` codificado $C$.
   * Se $C \le 0$: o valor é nulo (`None` / `null`).
   * Se $C > 0$: o comprimento da string é $L = C - 1$. Lê $L$ bytes UTF-8.

---

## 4. Estrutura do Snapshot Inicial da Sala (`Initial Room State`)

Imediatamente após a descompactação, o stream binário apresenta o snapshot de inicialização da sala:

```
[Sync Header] ──▶ [Metadados da Sala] ──▶ [Definição do Estádio] ──▶ [Match State] ──▶ [Players Roster] ──▶ [Team Colors]
```

### 4.1. Tabela de Sincronização (`Sync Header`)
* `sync_count`: `u16_be`.
* Para cada entrada em `sync_count`:
  * `delta_tick`: `varint`.
  * `sync_byte`: `u8`.

### 4.2. Metadados Básicos da Sala
* `room_name`: `nullable_str` (Nome público da sala).
* `locked`: `u8` (`0`: destravada, `1`: protegida por senha).
* `score_limit`: `i32_be` (Limite de gols para vitória; 0 = sem limite).
* `time_limit`: `i32_be` (Limite de tempo em minutos; 0 = sem limite).
* `r.read_i16_be()`, `r.read_u8()`, `r.read_u8()`: Flags de configuração da sala e trava de times.

### 4.3. Especificação Completa do Estádio (`Stadium Definition`)
O HaxBall verifica a tag do estádio:
* `stadium_tag`: `u8`.
  * Se `stadium_tag < 255`: Trata-se de um mapa padrão embutido do jogo:
    * `0`: Classic
    * `1`: Easy
    * `2`: Small
    * `3`: Big
    * `4`: Rounded
    * `5`: Hockey
    * `6`: Big Hockey
    * `7`: Big Easy
    * `8`: Big Rounded
    * `9`: Huge
  * Se `stadium_tag == 255`: Trata-se de um **estádio customizado serializado** (o arquivo `.hbs` completo em binário):
    1. `name`: `nullable_str` (Nome descritivo do mapa, ex.: `"Futsal 3x3 GLH by Bazinga"`).
    2. `bg_type`: `i32_be` (Tipo de fundo: `0`: normal, `1`: grama, `2`: hóquei).
    3. `width`, `height`: `f64_be` (Largura e altura útil do campo).
    4. `cam_w`, `cam_h`: `f64_be` (Limites de visão de câmera).
    5. `spawn_dist`: `f64_be` (Distância dos pontos de spawn dos jogadores em relação ao centro).
    6. `bg_color`: `i32_be` (Cor hexadecimal de fundo do gramado).
    7. **Física da Bola Padrão:** `radius` (f64), `invMass` (f64), `damping` (f64), `bCoef` (f64), `kickAcc` (f64), `kickDamping` (f64), `kickStrength` (f64).
    8. **Lista de Vértices (`num_verts`: u8):** Para cada vértice: $(x, y)$ em f64, `bCoef` em f64, máscaras de colisão `cMask` e `cGroup` em i32, `trait` em u8.
    9. **Lista de Segmentos (`num_segs`: u8):** Para cada segmento: vértices de origem e destino $(v_0, v_1)$ em u8, `bCoef` em f64, `curve` em f64 (curvatura da parede em graus), `color` em i32, máscaras de colisão e visibilidade.
    10. **Lista de Planos de Borda (`num_planes`: u8):** Vetor normal $(n_x, n_y)$ em f64, distância da origem em f64, `bCoef` e máscaras de colisão.
    11. **Lista de Gols (`num_goals`: u8):** Postes da trave $(p_{0x}, p_{0y}, p_{1x}, p_{1y})$ em f64 e identificador do time do gol (`team`: i8, onde `1: Red, 2: Blue`).
    12. **Lista de Discos de Física Extras (`num_discs`: u8):** Discos independentes no estádio com posições, velocidades, raios e massas.
    13. **Lista de Juntas Elásticas (`num_joints`: u8):** Restrições elásticas de distância entre discos.

### 4.4. Estado da Partida em Andamento (`has_match`)
* `has_match`: `u8` (`!= 0` se uma partida já estiver em progresso no instante exato do início da gravação).
* Se verdadeiro, desempacota o estado da bola e o tempo decorrido no cronômetro da rodada.

### 4.5. Lista de Jogadores Presentes (`Players Roster`)
* `num_players`: `u8`.
* Para cada jogador:
  * `player_id`: `i32_be` (Identificador numérico do jogador na sala).
  * `player_name`: `nullable_str` (Nickname do jogador).
  * `team`: `i8` (`0`: Espectador, `1`: Time Vermelho/Red, `2`: Time Azul/Blue).
  * `admin`: `i32_be` ou flag.
  * `country`: `nullable_str`.
  * `avatar`: `nullable_str`.
  * `has_disc`: `u8`. Se `1`, o jogador já possui um avatar de disco instanciado na física (inclui $x, y, vx, vy, radius$).

### 4.6. Cores dos Times (`Team Colors`)
Desempacota a configuração visual dos times Red e Blue:
* Ângulo das listras do uniforme (`angle`: i16_be).
* Cor do texto do avatar (`avatar_color`: i32_be).
* `num_colors`: `u8` e paleta de cores hexadecimais em inteiros de 32 bits.

---

## 5. Stream de Pacotes de Ações e Eventos (`Action Packets Stream`)

Após o estado inicial, o restante do arquivo (frequentemente 90% a 95% do volume de bytes) é uma sequência contínua de pacotes de eventos temporais até o final da gravação (`r.remaining() == 0`).

Cada pacote possui o formato:
```
[delta_ticks : varint]  ──▶  [action_type : u8]  ──▶  [payload específico da ação]
```

* **`delta_ticks`:** Número de ticks de simulação passados desde a ação anterior. O tick atual da partida é acumulado incrementalmente:
  $$\text{curr\_tick} \leftarrow \text{curr\_tick} + \text{delta\_ticks}$$
* **`action_type`:** Byte que define o identificador do evento.

### 5.1. Tabela Completa de Tipos de Ação (`Action IDs`)

| ID | Nome no Código Interno | Descrição do Evento | Estrutura dos Bytes Subsequentes |
|---|---|---|---|
| **0** | `jr` | **Mensagem de Chat** | Texto (`str_varint_len`), ID do autor (`i32_be`), flags (`u8`, `u8`). |
| **1** | `br` | **Atualização de Ping** | Valor do ping em ms (`u8`). |
| **2** | `Ai` | **Início de Partida / Kickoff** | Sem payload extra. Marca o apito inicial. |
| **3** | `wr` | **Input de Teclado (Jogador)** | Máscara de bits de input (`u32_be`). |
| **5** | `Cr` | **Entrada de Jogador (Join)** | ID (`i32_be`), Nome (`nullable_str`), País (`nullable_str`), Avatar (`nullable_str`). |
| **6** | `gr` | **Saída de Jogador (Leave)** | ID (`i32_be`), Motivo (`nullable_str`), Flag (`u8`). |
| **7** | `_` | **Arbitragem Instantânea** | Sem payload. |
| **8** | `_` | **Reinício de Posicionamento** | Sem payload. |
| **9** | `xr` | **Pausa / Despausa** | Flag booleana de estado de pausa (`u8`). |
| **10** | `Nr` | **Alteração de Regras** | Novo `score_limit` (`i32_be`), novo `time_limit` (`i32_be`). |
| **11** | `Er` | **Mudança Dinâmica de Estádio** | Estrutura completa de estádio desempacotada em tempo de execução. |
| **12** | `Dr` | **Mudança de Time de Jogador** | ID do jogador (`i32_be`), novo time (`i8`: 0=Spec, 1=Red, 2=Blue). |
| **13** | `_r` | **Trava de Times** | Flag de trava de times (`u8`). |
| **14** | `Pr` | **Privilégios de Administrador** | ID do jogador (`i32_be`), status de admin (`u8`). |
| **15** | `_` | **Reset de Cronômetro** | Sem payload. |
| **16** | `Ar` | **Gol Marcado** | Time que marcou o gol (`u8`: 1=Red, 2=Blue). |
| **18** | `Or` | **Alteração de Avatar** | Novo avatar textual (`nullable_str`). |

---

## 6. Mapeamento de Teclas e Bitmask de Input (`action_type == 3`)

Quando um pacote do tipo **`3`** é disparado, ele representa o envio de comandos do jogador local para o motor de física.

A máscara de bits (`input_mask`) compacta o estado simultâneo de todas as teclas:

```
Bit 0 (0x01) ──▶ CIMA (W / Seta Cima)
Bit 1 (0x02) ──▶ BAIXO (S / Seta Baixo)
Bit 2 (0x04) ──▶ ESQUERDA (A / Seta Esquerda)
Bit 3 (0x08) ──▶ DIREITA (D / Seta Direita)
Bit 4 (0x10) ──▶ CHUTE (Espaço / Tecla X / Ctrl / Shift)
```

### 6.1. Tradução para Vetores de Ação
* **Direcional X (`xdir`):**
  $$xdir = \begin{cases} -1, & \text{se } (\text{input\_mask} \ \& \ 0x04) \ne 0 \text{ e } (\text{input\_mask} \ \& \ 0x08) == 0 \\ +1, & \text{se } (\text{input\_mask} \ \& \ 0x08) \ne 0 \text{ e } (\text{input\_mask} \ \& \ 0x04) == 0 \\ 0, & \text{caso contrário} \end{cases}$$
* **Direcional Y (`ydir`):**
  $$ydir = \begin{cases} -1, & \text{se } (\text{input\_mask} \ \& \ 0x01) \ne 0 \text{ e } (\text{input\_mask} \ \& \ 0x02) == 0 \\ +1, & \text{se } (\text{input\_mask} \ \& \ 0x02) \ne 0 \text{ e } (\text{input\_mask} \ \& \ 0x01) == 0 \\ 0, & \text{caso contrário} \end{cases}$$
* **Chute (`kick`):**
  $$\text{kick} = (\text{input\_mask} \ \& \ 0x10) \ne 0$$

Essa representação encaixa-se diretamente no espaço de ação do nosso ambiente Gym (`gym_env/`) e na API Headless do HaxBall (`online_bridge/`).

---

## 7. Vulnerabilidades Comuns em Parsers e Dessincronização de Bytes

Ao construir decodificadores para `.hbr2`, três problemas clássicos causam quebra silenciosa da leitura:

1. **Dessincronização de 1 Byte em `nullable_str`:**
   Se uma string nula não for consumida adequadamente ou tiver seu código de comprimento lido como inteiro fixo em vez de `varint`, o leitor desloca o ponteiro por 1 ou 2 bytes. Como o formato não possui marcadores de alinhamento periódico, todas as leituras seguintes falham com `EOFError` ou `struct.error`.
2. **Estádios com Tag `255` Não Mapeados Integralmente:**
   Muitos parsers simples ignoram que um estádio customizado inclui arrays dinâmicos de vértices, segmentos e planos. Se o leitor pular um número fixo de bytes em vez de iterar sobre o número exato de elementos geométricos (`num_verts`, `num_segs`), ele cai em dados corrompidos.
3. **Quebra Precoce do Loop de Ações:**
   Parsers ingênuos encerram o processamento (`break`) ao encontrar qualquer byte de ação não documentado. O tratamento correto exige ler o pacote com segurança ou capturar exceções por bloco de frames.

---

## 8. Mapeamento do Ecossistema Global de Fontes e Dados

Além dos arquivos já raspados no `thehax.pl` e no canal de log do bot no Discord, o ecossistema mundial de HaxBall possui repositórios, ferramentas e plataformas de destaque:

### 8.1. Plataformas e APIs de Replays Automatizados

#### A. MrREPLAY / MrHOST Platform
* **Site / API:** [mrhosthaxball.com](https://mrhosthaxball.com/)
* **Descrição:** A maior plataforma de hospedagem de salas e geração automática de estatísticas do mundo (muito forte na América Latina e Europa). Salas que utilizam o MrHOST enviam automaticamente cada partida para a CDN do MrREPLAY.
* **API Endpoints:** Permite consulta de partidas por ID de sala, liga ou jogador com metadados estruturados em JSON e download direto do `.hbr2`.

#### B. Haxball Replay Analyzer
* **Repositório GitHub:** [`haxball-replay-analyzer/haxball-replay-analyzer.github.io`](https://github.com/haxball-replay-analyzer/haxball-replay-analyzer.github.io)
* **Aplicação Web:** [haxball-replay-analyzer.github.io](https://haxball-replay-analyzer.github.io/)
* **Destaques Técnicos:** Desenvolvido em React/JavaScript, contém o analisador mais robusto de estatísticas da comunidade (mapas de calor de posicionamento, porcentagem de posse nos terços do campo, passes certos e chutes a gol).

#### C. HaxMaps
* **Site:** [haxmaps.com](https://haxmaps.com/)
* **Descrição:** Repositório central mundial de mapas customizados (`.hbs`). Utilizado pelo nosso projeto para baixar e parametrizar a física exata das quadras oficiais de Futsal 3v3 (mapa 7899) e Futsal 5v5 (mapa 9362).

---

### 8.2. Ligas Competitivas Mundiais com Arquivos de Partidas

| Comunidade / Liga | Região | Formatos Principais | Repositório / Acesso |
|---|---|---|---|
| **cis-haxball.ru** | Leste Europeu / CIS | Futsal 3v3, 4v4 Big | Servidor de replay público com milhares de arquivos indexados por data. |
| **Global League Haxball (GLH)** | Internacional | Futsal 3x3 e 1x1 Futsal | Fóruns oficiais e servidores Discord com torneios de elite. |
| **American Haxball Association (AHA)** | América do Norte | 1v1 Elo Ladder, Big 3v3 | Torneios 1v1 tradicionais com gravação obrigatória de playoffs. |
| **HaxLeague / FM-Haxball** | Reino Unido / Europa | 4v4 Real Soccer, 3v3 | Fóruns clássicos com acervos históricos de partidas oficiais. |
| **EFRS / FBF / Haxlife / LHL** | América do Sul (BR, AR, UY) | Futsal 3v3, Real Soccer 4v4 | Canais dedicados de Discord onde capitães postam os arquivos `.hbr2` após cada rodada. |

---

### 8.3. Projetos Open-Source de Engenharia Reversa e Bibliotecas

1. **`haxball-replay-decoder` (Rust):**
   * Repositório: Disponível em [crates.io/crates/haxball-replay-decoder](https://crates.io/crates/haxball-replay-decoder).
   * Destaque: Implementação em Rust de alta performance para desempacotamento e inspeção de cabeçalhos sem overhead de interpretador.
2. **`hbr2-clips` (Node.js):**
   * Repositório: [`KvensOs/hbr2-clips`](https://github.com/KvensOs/hbr2-clips).
   * Destaque: Extrai automaticamente trechos de gols e lances capitais a partir dos eventos `action_type == 16`.
3. **`node-haxball` / HaxBall Headless Host API:**
   * Biblioteca Node.js que implementa o protocolo WebRTC do HaxBall, permitindo criar salas sem interface gráfica e gravar partidas programaticamente via `room.startRecording()` e `room.stopRecording()`.
4. **`jonnyynnoj/haxball-replay-parser` (PHP):**
   * Parser histórico para a especificação legada `.hbr` (Flash). Útil como referência de transição arquitetural.

---

## 9. Diretrizes de Uso no Pipeline de IA (RL e Imitação)

Com base nesta especificação técnica completa:

1. **Decodificação Confiável de Replays Reais:**
   O parser implementado em [`haxball/data/hbr2_parser.py`](file:///c:/Users/caihe/Documents/antigravity/agitated-hertz/haxball/data/hbr2_parser.py) utiliza exatamente as regras de *raw deflate*, leitura de geometrias completas e casamento de bits de input especificadas neste documento.
2. **Inspeção de Cabeçalho sem Falsos Positivos:**
   A filtragem de qualidade de replays deve **sempre desempacotar o nome do estádio na seção 4.3** (`tag == 255`, `name`) em vez de confiar no título do arquivo ou em palavras-chave frágeis como `"x1"` (que frequentemente indicam placar).
3. **Compatibilidade com Treinamento Online:**
   A bitmask de inputs descrita na Seção 6 garante que a saída gerada pela política treinada em PyTorch (`discrete_actions` ou `continuous_actions`) seja 100% intercambiável com os pacotes aceitos pelo cliente oficial do HaxBall.
