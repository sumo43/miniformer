rm *txt
rm -rf opus*
wget https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-en-es-v1.0.tar.gz
tar xvf *.tar.gz
mv opus-100-corpus/v1.0/supervised/en-es/* .
rm -rf opus-100-corpus

mv opus.en-es-dev.en en_dev.txt
mv opus.en-es-dev.es es_dev.txt

mv opus.en-es-test.en en_test.txt
mv opus.en-es-test.es es_test.txt

mv opus.en-es-train.en en_train.txt
mv opus.en-es-train.es es_train.txt
