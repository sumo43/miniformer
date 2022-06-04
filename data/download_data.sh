rm *en
rm *es
rm -rf opus*
wget https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-en-es-v1.0.tar.gz
tar xvf *.tar.gz
mv opus-100-corpus/v1.0/supervised/en-es/* .
rm -rf opus-100-corpus
