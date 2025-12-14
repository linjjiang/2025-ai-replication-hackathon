import os

from src.model.rnn import RNN
from src.task.retrocue_wm import RetrocueWMTask

def main():
    dataset = RetrocueWMTask(path=f'{os.environ.get("DATA_PATH")}/debug/retrocue_wm.pt')
    x, y = dataset[:10]
    model = RNN(input_size=x.shape[-1], hidden_size=200, output_size=y.shape[-1])

    out = model(x)
    print(out.shape)

    model.return_hidden = True
    out, hidden = model(x)
    print(out.shape, hidden.shape)

if __name__ == "__main__":
    main()

