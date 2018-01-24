# Projeto Final Curso de Verão

Neste projeto vamos capturar séries de preços de mercados de criptomoedas visualizá-los e armazená-los  para posterior uso.
Você pode fazer sua implementação diretamente em um notebook Jupyter. Este notebook deve ser mantido em um repositório público  no Github.

## Captura dos Preços
Vamos usar aqui a biblioteca [ccxt](https://github.com/ccxt/ccxt) para capturar a série de preços. Consulte a documentação deste pacote para descobrir como realizar o Download. Você pode consultar este [tutorial](https://github.com/fccoelho/crypto_algo_trading/blob/master/CCXT%20tutorial.ipynb) para ter uma breve introdução.

Você deve capturar dados `OHLCV`, ou seja dados de preços, agregados a intervalos de 5 minutos. Se não estiverem disponíveis tente `trades` usando o método `fetch_trades`

Você pode usar o código abaixo para verificar qual Exchange oferece dados `OHLCV`:

```python
import ccxt

for broker in ccxt.exchanges:
    try:
        exchange = getattr(ccxt, broker)()
        market = list(exchange.load_markets().keys())[0]
        exchange.fetch_ohlcv(market)
        print("\033[92m", broker, " Serves OHLCV data")
    except Exception as e:
        print("\033[91m", e)
        pass

```

Escolha uma exchange e informe ao professor. (Para evitar que todos escolham a mesma). Escolha ao menos dois mercados nesta exchange. Por exemplo: `DASH/USD` e `DASH/EUR`.

Se a `ccxt` não oferecer a captura que você deseja, mas você sabe que é possível obter os dados, implemente a função de captura em um arquivo `<nomedaexchange>.py` e faça um pull-request.

O seu codigo de captura deve consistir em uma função ou uma classe, capaz de capturar um intervalo de dados de pelo menos um ano, respeitando a taxa de requisições máxima de cada exchange, e que seja capaz de tratar exceções de captura e retomar a captura até completar a tarefa.

### Exchanges escolhidas
- Bittrex: Luis Henrique
- Cryptopia: Bianca Gonçalves Pereira
- CEX.IO: Artur Chiaperini Grover
- : Brenda Q. Prallon
- Poloniex: Guilherme Horta
- Kucoin: Marcelo Barata Ribeiro
- Kraken: Bernardo Bikman
- Binance: Marcelo Orgler
- Mercado Bitcoin: Luiz Claudio
- Foxbit: Luis Felipe Kopp
- Bitfinex: Daniel Carletti
- 1BTCXE: Alessandro Tessarollo
- GDAX: Igor Carvalho
- TIDEX: Franklin Oliveira
- Gemini: Felipe Santos
- Dash: Arthur José Quintão Silva
- BTCChina: Larissa Machado
- OKEX: Fernanda Pedrosa
- : Tomaz Leal
- Coincheck: Pedro Medeiros Teixeira
- Coinbase: Juliana de Araujo C.B. Castro
- Coinmarketcap: Luiz Bezina de O. Preto
- Huobi: Igor Sales do Nascimento
- Bithumb: Denise de Oliveira Alves Carneiro
- Bleutrade: Rafael Martins Kovashihara
- Bittrex: Pedro Issler **REPETIDA, TROCAR**
- THE ROCK: Daniel Quintão de Moraes
- HitBTC: João Vítor Amaro

## Visualização
Uma vez que já tenha implementado um capturador, construa visualizações interativas de seus dados. Voce pode consultar os notebooks deste repositório para se inspirar, mas precisará construir pelo menos **5 visualizações diferentes** que aplicará aos seus dados. Recomendo a utilização da Biblioteca Holoviews. Mas a escolha é livre.

## Armazenamento dos dados
Vamos agora realizar o armazenamento dos dados capturados. O dados que são devolvidos pelo ccxt são em formato JSON, logo será necessário manipular os dados para escrevê-lo em um banco de dados. VoCê tem duas opções de bancos de dados para utilizar: SQLite, banco relacional mais simples, e [Influxdb](https://www.influxdata.com/time-series-platform/influxdb/) juntamente com seu [cliente python](https://influxdb-python.readthedocs.io/en/latest/include-readme.html). O Influxdb é um banco de dados especializado em séries temporais facilitando análises sobre este tipo de dados. Os alunos que optarem por usar o Influxdb receberão crédito extra.

Uma vez implementado o armazenamento adapte seu código de visualização para visualizar os dados a partir do banco.
