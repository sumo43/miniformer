from utils.general import ModelParams, TransformerTrainer
from utils.data import load_data
from utils.preproc import preprocess_data, postprocess_data, to_string
from model.transformer import Transformer

"""

Eval 
For now, we take a sentence in english, and translate it word-by-word into spanish until it hits the max seq length limit
TODO add validation set
TODO add a way to keep the vocab as a separate file and load it into memory for translation

"""

mp = ModelParams(d_model=d_model, 
    h=h, 
    n_encoders=n_encoders, 
    n_decoders=n_decoders, 
    d_ff=d_ff,
    max_seq_length=max_seq_length,
    batch_size=batch_size
    )

inputs, outputs = load_data(mp)
in_ds, out_ds, eng_vocab, sp_vocab = preprocess_data(mp, inputs, outputs, test=True)

data = (in_ds, out_ds, eng_vocab, sp_vocab)

device = mp.device

#t = SimpleFormer(mp)
t = Transformer(mp).to(device)
    
ev = TransformerEvaluator(mp, t, data)

ev.eval()