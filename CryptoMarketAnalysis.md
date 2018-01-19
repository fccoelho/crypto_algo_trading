# Projeto Final Curso de Verão

Neste projeto vamos capturar séries de preços de mercados de criptomoedas visualizá-los e armazená-los  para posterior uso.
Você pode fazer sua implementação diretamente em um notebook Jupyter. Este notebook deve ser mantido em um repositório público  no Github.

## Captura dos Preços
Vamos usar aqui a biblioteca [ccxt](https://github.com/ccxt/ccxt) para capturar a série de preços. Consulte a documentação deste pacote para descobrir como realizar o Download. Você pode consultar este [tutorial](https://github.com/fccoelho/crypto_algo_trading/blob/master/CCXT%20tutorial.ipynb) para ter uma breve introdução.

Você deve capturar `ticker data`, ou seja dados de preços instantâneos.

Escolha uma exchange e informe ao professor. (Para evitar que todos escolham a mesma). Escolha ao menos dois mercados nesta exchange. Por exemplo: `DASH/USD` e `DASH/EUR`

O seu codigo de captura deve consistir em uma função ou uma classe, capaz de capturar um intervalo de dados de pelo menos um ano, respeitando a taxa de requisições máxima de cada exchange, e que seja capaz de tratar exceções de captura e retomar a captura até completar a tarefa.

## Visualização
Uma vez que já tenha implementado um capturador, construa visualizações interativas de seus dados. Voce pode consultar os notebooks deste repositório para se inspirar, mas precisará construir pelo menos **5 visualizações diferentes** que aplicará aos seus dados. Recomendo a utilização da Biblioteca Holoviews. Mas a escolha é livre.

## Armazenamento dos dados
Vamos agora realizar o armazenamento dos dados capturados. O dados que são devolvidos pelo ccxt são em formato JSON, logo será necessário manipular os dados para escrevê-lo em um banco de dados. VoCê tem duas opções de bancos de dados para utilizar: SQLite, banco relacional mais simples, e [Influxdb](https://www.influxdata.com/time-series-platform/influxdb/) juntamente com seu [cliente python](https://influxdb-python.readthedocs.io/en/latest/include-readme.html).

Uma vez implementado o armazenamento adapte seu código de visualização para visualizar os dados a partir do banco.
