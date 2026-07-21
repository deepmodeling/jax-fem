import os
from jax_fem.utils import make_video

data_dir = os.path.join(os.path.dirname(__file__), 'data') 


if __name__=="__main__":
    make_video(data_dir)
